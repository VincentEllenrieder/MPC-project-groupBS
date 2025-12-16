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

        #Q = np.eye(self.nx)
        #R = np.eye(self.nu)

        # allow subsystem-specific tunings
        Q = getattr(self, "Q", np.eye(self.nx))
        R = getattr(self, "R", np.eye(self.nu))

        K, P, _ = dlqr(self.A, self.B, Q, R)

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
        cost += cp.quad_form(x[:, self.N], P)

        # Constraints
        constraints = []
        u_min_abs = self.LBU[self.u_ids]
        u_max_abs = self.UBU[self.u_ids]
        x_min_abs = self.LBX[self.x_ids]
        x_max_abs = self.UBX[self.x_ids]

        # Add small margin to avoid numerical issues
        margin_u = 1e-3
        margin_x = 1e-3

        u_min = (u_min_abs - self.us) + margin_u
        u_max = (u_max_abs - self.us) - margin_u
        x_min = (x_min_abs - self.xs) + margin_x
        x_max = (x_max_abs - self.xs) - margin_x

        # Compute terminal invariant set Xf
        # Define initial set X0 as polyhedron Hx x <= hx
        # X0 for invariant set computation only use finite bounds
        Hx_list = []
        hx_list = []

        for i in range(self.nx):
            if np.isfinite(x_max[i]):
                e_i = np.zeros(self.nx)
                e_i[i] = 1.0 # this picks out x_i
                Hx_list.append(e_i)
                hx_list.append(x_max[i])
            if np.isfinite(x_min[i]):
                e_i = np.zeros(self.nx)
                e_i[i] = -1.0 # corresponds to -x_i
                Hx_list.append(e_i)
                hx_list.append(-x_min[i])

        Hx = np.vstack(Hx_list) if Hx_list else np.zeros((0, self.nx))
        hx = np.array(hx_list)   if hx_list else np.zeros((0,))

        Hu = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        hu = np.hstack([u_max, -u_min])

        Hx_u = -Hu @ K
        hx_u = hu

        H0 = np.vstack([Hx, Hx_u]) # (2*nx + 2*nu) x nx
        h0 = np.hstack([hx, hx_u]) # (2*nx + 2*nu)

        X0 = Polyhedron.from_Hrep(H0, h0)

        #print("H0 shape:", H0.shape, "h0 shape:", h0.shape)
        #print("X0 dim:", X0.dim)
        #print("X0 empty?", X0.is_empty)

        A_cl = self.A - self.B @ K
        self.Xf = self._max_invariant_set(A_cl, X0)
      
        # Constraints over the horizon
        for k in range(self.N):
            # System dynamics constraints
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]
            # Input constraints
            constraints += [u_min <= u[:, k], u[:, k] <= u_max]
            # State constraints
            constraints += [x_min <= x[:, k], x[:, k] <= x_max]
        # Init state constraint    
        constraints += [x[:, 0] == self.x0_param]
        
        # Terminal constraint x_N ∈ X_f
        Ff, ff = self.Xf.A, self.Xf.b
        constraints += [Ff @ x[:, self.N] <= ff]

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
        x0_sub = x0 # no slicing
        self.x0_param.value = x0_sub - self.xs

        # Verbose solver output for debugging
        self.ocp.solve(
            solver=cp.OSQP,
            warm_start=True,
            max_iter=20000,
            eps_abs=1e-4,
            eps_rel=1e-4,
        )
        
        #self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"MPC QP infeasible or failed, status = {self.ocp.status}")


        u0_delta = self.u[:, 0].value
        u0 = self.us + u0_delta

        x_traj = self.x.value + self.xs[:, None]
        u_traj = self.u.value + self.us[:, None]

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    @staticmethod
    def _max_invariant_set(A_cl, X: Polyhedron, max_iter = 20) -> Polyhedron:
        """
        Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
        """
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            # O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            preO = Polyhedron.from_Hrep(F @ A_cl, f)
            O = preO.intersect(O)
            
            if O == Oprev:
                converged = True
                break
            print('Iteration {0}... not yet converged\n'.format(itr))
            itr += 1
        
        if converged:
            print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
        return O

    
