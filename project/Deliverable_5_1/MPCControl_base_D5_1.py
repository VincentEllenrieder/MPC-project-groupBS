import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from itertools import combinations


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
    UBU : np.ndarray
    LBU : np.ndarray
    UBX : np.ndarray
    LBX : np.ndarray

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

        # Bounds definition
        UBU = np.array([np.deg2rad(15), np.deg2rad(15), 80, 20])
        LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])
        UBX = np.array([np.inf,  np.inf,  np.inf,
                        np.deg2rad(10),  np.deg2rad(10),  np.inf,
                        np.inf,  np.inf,  np.inf,
                        np.inf,  np.inf, np.inf])
        LBX = np.array([-np.inf, -np.inf, -np.inf,
                        -np.deg2rad(10), -np.deg2rad(10), -np.inf,
                        -np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, 0.0])
        self.UBU = UBU[self.u_ids]
        self.LBU = LBU[self.u_ids]
        self.UBX = UBX[self.x_ids]
        self.LBX = LBX[self.x_ids]

        self.U = Polyhedron.from_bounds(self.LBU, self.UBU)
        self.X = Polyhedron.from_bounds(self.LBX, self.UBX)

        FULL_STATE_NAMES = [
            r'$\omega_x$', r'$\omega_y$', r'$\omega_z$',
            r'$\alpha$',   r'$\beta$',   r'$\gamma$',
            r'$v_x$',      r'$v_y$',      r'$v_z$',
            r'$x$',        r'$y$',        r'$z$']

        self.red_state_names = [FULL_STATE_NAMES[i] for i in self.x_ids]

        # Cache for terminal set to avoid computing it if target same as for previous call of get_u
        self._cached_xt = None
        self._cached_ut = None
        self._cached_Xf = None

        self._setup_controller() # this method is called at initialization of the controller -> the optimization problem and its variables are all acessible

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        Q = getattr(self, "Q", np.eye(self.nx))
        R = getattr(self, "R", np.eye(self.nu))

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_par = cp.Parameter((self.nx,), name='x0')
        self.xt_par = cp.Parameter((self.nx,), name='xtarget')
        self.ut_par = cp.Parameter((self.nu,), name= 'utarget')
        
        x_diff = self.x_var - cp.reshape(self.xt_par, (self.nx, 1))
        u_diff = self.u_var - cp.reshape(self.ut_par, (self.nu, 1))

        # Add Slack
        self.eps = cp.Variable((self.nx, self.N), nonneg=True)
        rho = getattr(self, "rho_slack", 1e4)

        x_min = self.LBX
        x_max = self.UBX

        #only for the 3-state subsystems (xvel, yvel), soften the second reduced state (index 1).
        # In your xvel subsystem, x_ids = [1, 4, 6] → reduced state index 1 corresponds to global state 4 = β.
        # In yvel, x_ids = [0, 3, 7] → reduced index 1 corresponds to global state 3 = α.
        soft = np.zeros(self.nx)
        if self.nx == 3:          # xvel / yvel
            soft[1] = 1.0         # alpha/beta component

        S = np.diag(soft)  # nx x nx

        # Costs (objective function)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(x_diff[:,k], Q)
            cost += cp.quad_form(u_diff[:,k], R)

            # penalize slack (only on softened component)
            cost += rho * cp.sum_squares(cp.multiply(soft, self.eps[:, k]))
        
        # Terminal cost
        K, P, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        cost += cp.quad_form(x_diff[:, -1], P)        

        # System (equality) constraint
        constraints = []
        constraints.append(self.x_var[:, 0] == self.x0_par)
        constraints.append(self.A @ (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, :-1] + self.B @ (self.u_var - cp.reshape(self.us, (self.nu, 1))) == (self.x_var - cp.reshape(self.xs, (self.nx, 1)))[:, 1:]) # x^+ - xs = A(x-xs) + B(u-us)

        # Inequality constraints
        #Need to remove, otherwise overwritten the soft
        #constraints.append(self.X.A @ self.x_var <= self.X.b.reshape(-1, 1)) # x in X for all k = 0, ..., N

        constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1)) # u in U for all k = 0, ..., N-1

        # Hard bounds for everyone, except softened indices get slack
        # applies to all k = 0..N-1
        constraints.append(self.x_var[:, :-1] <= x_max[:, None] + S @ self.eps)
        constraints.append(self.x_var[:, :-1] >= x_min[:, None] - S @ self.eps)

        # Terminal constraint
        X_delta = Polyhedron.from_bounds(self.LBX - self.xs, self.UBX - self.xs)
        U_delta = Polyhedron.from_bounds(self.LBU - self.us, self.UBU - self.us)

        A_cl = self.A + self.B @ K

        KU_delta = Polyhedron.from_Hrep(U_delta.A @ K, U_delta.b)
        Omega = X_delta.intersect(KU_delta)
        
        i = 0
        max_iter = 50
        while i < max_iter :
            H, h = Omega.A, Omega.b
            pre_omega = Polyhedron.from_Hrep(H @ A_cl, h)
            Omega_new = pre_omega.intersect(Omega)
            Omega_new.minHrep(True)
            _ = Omega_new.Vrep 
            if Omega == Omega_new :
                Omega = Omega_new
                print("Maximum invariant set found after {0} iterations !\n" .format(i+1))
                break
            print("Not yet convgerged at iteration {0}" .format(i+1))
            Omega = Omega_new
            i += 1

        self.X_f = Omega
        # If empty terminal set -> infeasible
        if self.X_f.Vrep.V.size == 0:
            raise RuntimeError(
                f"[{self.subsys_name}] Terminal set is EMPTY around current target; QP will be infeasible."
            )
        
        constraints.append(self.X_f.A @ x_diff[:, -1] <= self.X_f.b) # x_N - x_t in Xf
        constraints.append(self.U.A @ (self.ut_par + K @ x_diff[:, -1]) <= self.U.b) # u_N = u_t + K (x_N - x_t) in U

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

        # Initialize QP parameters
        self.x0_par.value = x0
        if x_target is None:
            self.xt_par.value = self.xs
        else:
            self.xt_par.value = x_target
        if u_target is None:
            self.ut_par.value = self.us
        else:
            self.ut_par.value = u_target

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

        # YOUR CODE HERE
        #################################################

    
