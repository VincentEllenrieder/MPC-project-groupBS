import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int
    Q: np.ndarray
    R: np.ndarray
    subsys_name: str
    UBX: np.ndarray
    LBX: np.ndarray
    UBU: np.ndarray
    LBU: np.ndarray
    X: Polyhedron
    U: Polyhedron


    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self.UBU = np.array([np.deg2rad(15), np.deg2rad(15), 80, 20])
        self.LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])
        self.UBX = np.array([np.inf,  np.inf,  np.inf,
                        np.deg2rad(10),  np.deg2rad(10),  np.inf,
                        np.inf,  np.inf,  np.inf,
                        np.inf,  np.inf, np.inf])
        self.LBX = np.array([-np.inf, -np.inf, -np.inf,
                        -np.deg2rad(10), -np.deg2rad(10), -np.inf,
                        -np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, 0.0])
        self.UBU = self.UBU[self.u_ids]
        self.LBU = self.LBU[self.u_ids]
        self.UBX = self.UBX[self.x_ids]
        self.LBX = self.LBX[self.x_ids]
        print(f"Subsystem {self.subsys_name}: \n")
        print(f"Input upper bounds are {self.UBU}")
        print(f"Input lower bounds are {self.LBU}")
        print(f"State upper bounds are {self.UBX}")
        print(f"State lower bounds are {self.LBX} \n")

        self.U = Polyhedron.from_bounds(self.LBU, self.UBU)
        self.X = Polyhedron.from_bounds(self.LBX, self.UBX)

        FULL_STATE_NAMES = [
            r'$\omega_x$', r'$\omega_y$', r'$\omega_z$',
            r'$\alpha$',   r'$\beta$',   r'$\gamma$',
            r'$v_x$',      r'$v_y$',      r'$v_z$',
            r'$x$',        r'$y$',        r'$z$']

        self.red_state_names = [FULL_STATE_NAMES[i] for i in self.x_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        
        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_par = cp.Parameter((self.nx,), name='x0')
        # Add Slack
        if self.nx == 4:
            self.eps_var = cp.Variable((1, self.N + 1), nonneg=True, name="epsilon") # eps => 0 incorporated here
            S = np.array([[1e4]])
            # rho = getattr(self, "rho_slack", 1e4) # needed ?
        else :
            self.eps_var = np.zeros((1, self.N + 1))
            S = np.array([[0]])
        
        x_diff = self.x_var - cp.reshape(self.xs, (self.nx, 1))
        u_diff = self.u_var - cp.reshape(self.us, (self.nu, 1))

        # Costs (objective function)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x_diff[:,k], self.Q)
            cost += cp.quad_form(u_diff[:,k], self.R)

            # penalize slack (only on softened components -> S is diagonal)
            cost += cp.quad_form(self.eps_var[:, k], S)
        
        # Terminal cost
        _, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        cost += cp.quad_form(x_diff[:, -1], P)  
        cost += cp.quad_form(self.eps_var[:, -1], S)      

        # System (equality) constraint
        constraints = []
        constraints.append(self.x_var[:, 0] == self.x0_par)
        constraints.append(self.A @ (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, :-1] + self.B @ (self.u_var - cp.reshape(self.us, (self.nu, 1))) == (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, 1:]) # x^+ - xs = A(x-xs) + B(u-us)

        # Inequality constraints

        constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1)) # u in U for all k = 0, ..., N-1

        # Hard bounds for everyone, except softened indices get slack
        # applies to all k = 0...N
        constraints.append(self.X.A @ self.x_var <= self.X.b.reshape(-1, 1) + np.ones((self.X.A.shape[0], 1)) @ (self.eps_var)) # x in X + eps for all k = 0, ..., N

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        self.x0_par.value = x0

        #self.ocp.solve()
        self.ocp.solve(
            solver=cp.PIQP,
            warm_start=True,
            max_iter=20000,
            eps_abs=1e-4,
            eps_rel=1e-4
        )
        
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP problem failed: {self.ocp.status}")
        
        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value

        return u0, x_traj, u_traj
