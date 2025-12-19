import numpy as np
import cvxpy as cp

from MPCControl_base_D5_1 import MPCControl_base
from control import dlqr


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg
    tracked_idx = 0   # only state is v_z

class MPCControl_zvel_tuned_default(MPCControl_zvel):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 5.0
    dP_max = min(80 - 66.7, 66.7 - 40.0)  # m/s per control step

    Q = np.array([[1/(vz_max**2)]])
    R = np.array([[1/(dP_max**2)]])

class MPCControl_zvel_tuned_final(MPCControl_zvel):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    def __init__(self, A, B, xs, us, Ts, H):
        self.x_hat = np.zeros(self.nx)
        self.d_hat = np.zeros(self.nu)
        self.u_0 = np.zeros(self.nu)
        self.L = np.zeros((2,1))

        #Observer
        # x^+ = A x + B u + B d
        # d^+ = d
        # y   = 1 x + 0 d 

        # A_hat = [A  Bd;  0 I] = [A  B; 0  I]
        ny = self.nu
        nd = ny

        self.A_hat = np.vstack((
            np.hstack((self.A, self.B)),
            np.hstack((np.zeros((ny, self.nx)), np.eye(ny))))
        )

        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((nd, self.nu))))

        # C_hat = [C  Cd] = [1 0]
        self.C_hat = np.hstack((np.eye(ny,nd), np.zeros((ny,nd))))

        super().__init__(A, B, xs, us, Ts, H)

    
    def _setup_controller(self) -> None:

        vz_max = 5.0
        dP_max = min(80 - 66.7, 66.7 - 40.0)  # m/s per control step

        self.Q = np.array([[20 * (1/(vz_max**2))]])
        self.R = np.array([[(1/(dP_max**2)) / 2]])

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_par = cp.Parameter((self.nx,), name='x0')
        self.xt_par = cp.Parameter((self.nx,), name='xtarget')
        self.ut_par = cp.Parameter((self.nu,), name= 'utarget')
        
        x_diff = self.x_var - cp.reshape(self.xt_par, (self.nx, 1))
        u_diff = self.u_var - cp.reshape(self.ut_par, (self.nu, 1))

        # Costs (objective function)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x_diff[:,k], self.Q)
            cost += cp.quad_form(u_diff[:,k], self.R)
        
        # Terminal cost
        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        cost += cp.quad_form(x_diff[:, -1], P)        

        # System (equality) constraint
        constraints = []
        constraints.append(self.x_var[:, 0] == self.x0_par)
        constraints.append(self.A @ (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, :-1] + self.B @ (self.u_var - cp.reshape(self.us, (self.nu, 1))) == (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, 1:]) # x^+ - xs = A(x-xs) + B(u-us)

        # Inequality constraints
        #constraints.append(self.X.A @ self.x_var <= self.X.b.reshape(-1, 1)) # x in X for all k = 0, ..., N

        #constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1)) # u in U for all k = 0, ..., N-1
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # Place observer poles
        poles = np.array([0.8, 0.7])
        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = -res.gain_matrix.T


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        r = self.xs if x_target is None else x_target
        
        '''
        A = self.A
        B = self.B
        nx = self.nx
        nu = self.nu
        ny = nu
        nd = nu
        '''

        # x0 is the measurement

        #target selection, Compute steady state

        ny = self.nu
        nd = ny

        M = np.block([
        [np.eye(self.nx) - self.A,  -self.B],
        [np.eye(ny), np.zeros((ny, self.nu))]])
       
        rhs_top = self.B @ self.d_hat.reshape(-1,1)
        rhs_bot = np.array(r).reshape(-1,1) - self.d_hat.reshape(-1,1)
        rhs = np.vstack((rhs_top, rhs_bot))

        solution = np.linalg.solve(M, rhs)
        xs = solution[:self.nx].flatten()
        us = solution[self.nx:].flatten()

        # Initialize QP parameters
        self.xt_par.value = xs
        self.ut_par.value = us

        x_hat_aug = np.concatenate((self.x_hat, self.d_hat))

        tmp = self.A_hat @ x_hat_aug + self.B_hat @ self.u_0 + self.L @ (self.C_hat @ x_hat_aug - x0)

        self.x_hat = tmp[:self.nx]
        self.d_hat = tmp[self.nx:]

        self.x0_par.value = self.x_hat #estimator for the intial state

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

        self.u_0 = u0

        return u0, x_traj, u_traj




