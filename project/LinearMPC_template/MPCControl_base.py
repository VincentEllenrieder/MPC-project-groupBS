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

        # Constraints
        self.LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])
        self.UBU = np.array([ np.deg2rad(15),  np.deg2rad(15), 80.0,  20.0])

        self.LBX = np.array([
            -np.inf, -np.inf, -np.inf,
            -np.deg2rad(10), -np.deg2rad(10), -np.inf,
            -np.inf, -np.inf, -np.inf,
            -np.inf, -np.inf, 0.0
        ])
        self.UBX = np.array([
             np.inf,  np.inf,  np.inf,
             np.deg2rad(10),  np.deg2rad(10),  np.inf,
             np.inf,  np.inf,  np.inf,
             np.inf,  np.inf,  np.inf
        ])

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        self.x0_param = cp.Parameter(self.nx)
        self.x = cp.Variable((self.nx, self.N+1))  # columns: x_0 ... x_N
        self.u = cp.Variable((self.nu, self.N))    # columns: u_0 ... u_{N-1}

        x = self.x
        u = self.u

        Q = np.eye(self.nx)
        R = np.eye(self.nu)

        K, P, _ = dlqr(self.A, self.B, Q, R)

        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
        cost += cp.quad_form(x[:, self.N], P)

        constraints = []
        u_min_abs = self.LBU[self.u_ids]
        u_max_abs = self.UBU[self.u_ids]
        x_min_abs = self.LBX[self.x_ids]
        x_max_abs = self.UBX[self.x_ids]

        u_min = u_min_abs - self.us
        u_max = u_max_abs - self.us
        x_min = x_min_abs - self.xs
        x_max = x_max_abs - self.xs

        for k in range(self.N):
            # System dynamics constraints
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]
            # Input constraints
            constraints += [u_min <= u[:, k], u[:, k] <= u_max]
            # State constraints
            constraints += [x_min <= x[:, k], x[:, k] <= x_max]
        # Init state constraint    
        constraints += [x[:, 0] == self.x0_param]
        # Terminal state constraint
        constraints += [x_min <= x[:, self.N], x[:, self.N] <= x_max]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

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
        #################################################
        # YOUR CODE HERE
        x0_sub = x0[self.x_ids]
        self.x0_param.value = x0_sub - self.xs
        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        u0_delta = self.u[:, 0].value
        u0 = self.us + u0_delta

        x_traj = self.x.value
        u_traj = self.u.value

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
