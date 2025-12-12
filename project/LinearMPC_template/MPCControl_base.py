import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
from mpt4py import Polyhedron


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
        # This is overridden by the subclasses (xvel, yvel, etc.)
        pass

    def _build_problem(self, Q, R, P, u_min, u_max, x_min=None, x_max=None, term_set=None):
        """
        Helper method to construct the generic MPC problem.
        Called by subclasses in _setup_controller().
        """
        # 1. Define Variables
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x_init = cp.Parameter(self.nx)
        self.x_ref = cp.Parameter(self.nx) # Target state (for tracking)

        # Initialize Parameters
        self.x_ref.value = np.zeros(self.nx)

        # 2. Constraints & Cost
        cost = 0
        constraints = [self.x_var[:, 0] == self.x_init]

        for k in range(self.N):
            # Dynamics: x_k+1 = A*x_k + B*u_k
            constraints.append(self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k])

            # Input Constraints (u_k is delta_u)
            constraints.append(self.u_var[:, k] <= u_max)
            constraints.append(self.u_var[:, k] >= u_min)

            # State Constraints (if provided)
            if x_min is not None:
                constraints.append(self.x_var[:, k] >= x_min)
            if x_max is not None:
                constraints.append(self.x_var[:, k] <= x_max)

            # Cost: (x-ref)'Q(x-ref) + u'Ru
            state_err = self.x_var[:, k] - self.x_ref
            cost += cp.quad_form(state_err, Q) + cp.quad_form(self.u_var[:, k], R)

        # 3. Terminal Components
        term_err = self.x_var[:, self.N] - self.x_ref
        cost += cp.quad_form(term_err, P) # Terminal Cost

        if term_set is not None:
            # Terminal Set: Ax <= b
            # Note: Polyhedron.A and .b are used here
            constraints.append(term_set.A @ term_err <= term_set.b)

        # 4. Create Problem
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
        # 1. Update Parameters
        # x0 passed in is full state; we need to reduce it and subtract trim

        #x_sub = x0[self.x_ids] - self.xs
        #self.x_init.value = x_sub

        self.x_init.value = x0 - self.xs

        if x_target is not None:
            # For tracking: Target relative to trim
            #self.x_ref.value = x_target[self.x_ids] - self.xs
            self.x_ref.value = x_target - self.xs
        else:
            self.x_ref.value = np.zeros(self.nx)

        # 2. Solve
        # Warm start helps performance in loops
        self.ocp.solve(solver=cp.CLARABEL, warm_start=True) 
        # Note: Use OSQP or CLARABEL or ECOS

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print("chacal ça marche pas")
            # Fallback if solver fails
            u_opt = np.zeros(self.nu)
        else:
            u_opt = self.u_var[:, 0].value

        # 3. Return full absolute values
        # Add trim back to input
        u0 = u_opt + self.us

        # Return trajectories for plotting (add trim back)
        if self.x_var.value is not None:
            x_traj = self.x_var.value + self.xs.reshape(-1, 1)
            u_traj = self.u_var.value + self.us.reshape(-1, 1)
        else:
            x_traj = np.zeros((self.nx, self.N+1))
            u_traj = np.zeros((self.nu, self.N))

        return u0, x_traj, u_traj



    @staticmethod
    def max_invariant_set(A_cl, X: Polyhedron, max_iter = 20) -> Polyhedron:
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
            O = Polyhedron.from_Hrep(F @ A_cl, f).intersect(O)
            if O == Oprev:
                converged = True
                break
            print('Iteration {0}... not yet converged\n'.format(itr))
            itr += 1
        
        if converged:
            print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
        return O

    