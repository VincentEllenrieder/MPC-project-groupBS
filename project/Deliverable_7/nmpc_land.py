import numpy as np
import casadi as ca
from typing import Tuple
from control import dlqr
from src.rocket import cont2discrete



class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    Q = np.eye(12)
    R = np.eye(4)

    def __init__(self, rocket, H, xs, us):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        self.rocket = rocket
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]

        self.Ts = rocket.Ts
        self.H  = H
        self.N  = int(H/self.Ts)

        self.xs = xs
        self.us = us

        # Bounds definition
        self.d_lim = np.deg2rad(15)
        self.Pavg_min = 40.0
        self.Pavg_max = 80.0
        self.Pdiff_lim = 20.0
        self.beta_lim = np.deg2rad(80)

        self.Q = getattr(self, "Q", np.eye(12))
        self.R = getattr(self, "R", np.eye(4))

        # Terminal cost from discrete LQR at (xs, us)
        A, B = rocket.linearize(xs, us)
        Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(12), np.zeros((12,4))), self.Ts)
        _, P, _ = dlqr(Ad, Bd, self.Q, self.R)
        self.P = P

        self._X_init = None
        self._U_init = None

        self._setup_controller()

    def _setup_controller(self) -> None:
        opti = ca.Opti()
        N  = self.N
        Ts = self.Ts

        # Decision variables
        X = opti.variable(12, N + 1)    # states
        U = opti.variable(4, N)         # inputs
        X0 = opti.parameter(12) 

        # Constraints

        # Initial condition constraint
        opti.subject_to(X[:,0] == X0) 

        # Input constraints
        for k in range(N):
            opti.subject_to(opti.bounded(-self.d_lim, U[0, k], self.d_lim))      # δ1
            opti.subject_to(opti.bounded(-self.d_lim, U[1, k], self.d_lim))      # δ2
            opti.subject_to(opti.bounded(self.Pavg_min, U[2, k], self.Pavg_max)) # P_avg
            opti.subject_to(opti.bounded(-self.Pdiff_lim, U[3, k], self.Pdiff_lim)) # P_diff

        # State constraints
        # Beta constraint Singularity Avoidance and Ground constraint
        for k in range(N+1):
            opti.subject_to(opti.bounded(-self.beta_lim, X[4, k], self.beta_lim))
            opti.subject_to(X[11, k] >= 0.0)

        # Dynamics constaraints
        for k in range(N):
            x_next = self.rk4(self.f, X[:, k], U[:, k], Ts)
            opti.subject_to(X[:, k+1] == x_next)

       # Cost function stage + terminal
        cost = 0

        for k in range(N):
            dx = X[:,k] - self.xs
            du = U[:, k] - self.us
            cost += ca.mtimes([dx.T, self.Q, dx]) + ca.mtimes([du.T, self.R, du])

        dxN = X[:,N] -self.xs
        cost += ca.mtimes([dxN.T, self.P, dxN])   # terminal cost
        opti.minimize(cost)

        opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        self.opti, self.X, self.U, self.X0 = opti, X, U, X0

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.opti.set_value(self.X0, x0)

        # # initial guess helps a lot
        # self.opti.set_initial(self.X, np.tile(x0.reshape(12,1), (1, self.N+1)))
        # self.opti.set_initial(self.U, np.tile(self.us.reshape(4,1), (1, self.N)))

        if self._X_init is None:
            # First call → cold start
            self.opti.set_initial(self.X, np.tile(x0.reshape(12,1), (1, self.N+1)))
            self.opti.set_initial(self.U, np.tile(self.us.reshape(4,1), (1, self.N)))
        else:
            # Warm start: shifted solution
            self.opti.set_initial(self.X, self._X_init)
            self.opti.set_initial(self.U, self._U_init)

        sol = self.opti.solve()

        X_sol = sol.value(self.X)
        U_sol = sol.value(self.U)

        # Shift state guess
        X_init = np.zeros_like(X_sol)
        X_init[:, :-1] = X_sol[:, 1:]
        X_init[:, -1] = X_sol[:, -1]     # repeat terminal state

        # Shift input guess
        U_init = np.zeros_like(U_sol)
        U_init[:, :-1] = U_sol[:, 1:]
        U_init[:, -1] = U_sol[:, -1]     # repeat terminal input

        self._X_init = X_init
        self._U_init = U_init

        # x_ol = sol.value(self.X)
        # u_ol = sol.value(self.U)
        # u0 = u_ol[:,0]
        t_ol = t0 + np.arange(self.N+1)*self.Ts

        # return u0, x_ol, u_ol, t_ol
        return U_sol[:,0], X_sol, U_sol, t_ol
    
    @staticmethod
    def rk4(f, x, u, h):
        k1 = h * f(x, u)
        k2 = h * f(x + k1 / 2, u)
        k3 = h * f(x + k2 / 2, u)
        k4 = h * f(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
class NmpcCtrl_Tuned1(NmpcCtrl):


    wz_max    = np.deg2rad(120)
    gamma_max = np.deg2rad(90)
    Pdiff_max = 20.0

    beta_max = np.deg2rad(80)
    d2_max   = np.deg2rad(15)
    wy_max   = np.deg2rad(60)
    vx_max   = 5.0

    alpha_max = np.deg2rad(90)
    d1_max   = np.deg2rad(15)
    wx_max = np.deg2rad(60)
    vy_max = 5.0

    vz_max = 10.0
    P_max = 80

    x_max = 100
    y_max = 100
    z_max = 200

    Q = np.diag([  # 12 entries
        1 * (1/(wx_max**2)), 1 * (1/(wy_max**2)), 1 * (1/(wz_max**2)),              # wx wy wz
        1 * (1/(alpha_max**2)), 1 * (1/(beta_max**2)), 1 * (1/(gamma_max**2)),      # alpha beta gamma
        1 * (1/(vx_max**2)), 1 * (1/(vy_max**2)), 1 * (1/(vz_max**2)),              # vx vy vz
        1 * (1/(x_max**2)), 1 * (1/(y_max**2)), 1 * (1/(z_max**2))                  # x y z
    ])

    Q = Q.copy()
    Q[11,11] *= 8000   # z
    Q[9,9]   *= 550    # x
    Q[10,10] *= 550    # y
    Q[6,6]   *= 0.95   # vx
    Q[7,7]   *= 0.95   # vy
    Q[8,8]   *= 2      # vz small to allow dive. (note vz_max > vx_max = vy_max3)

    R = np.diag([1 * (1/(d1_max**2)), 1 * (1/(d2_max**2)), 1 * (1/(P_max**2)), 1 * (1/(Pdiff_max**2))])


