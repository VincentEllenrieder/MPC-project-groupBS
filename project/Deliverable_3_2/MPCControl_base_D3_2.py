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

        # Track last reference used to build Xf
        self._last_xref = None
        self._last_uref = None
        self._last_H = None
        self._last_h = None

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        Q = getattr(self, "Q", np.eye(self.nx))
        R = getattr(self, "R", np.eye(self.nu))

        # Terminal LQR
        K, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = K
        self.P = P

        # Parameters
        self.dx0_param = cp.Parameter(self.nx)   # x0 - x_ref
        self.xref_param = cp.Parameter(self.nx)  # x_ref (reduced)
        self.uref_param = cp.Parameter(self.nu)  # u_ref (reduced)

        # Decision vars (delta)
        dx = cp.Variable((self.nx, self.N + 1))
        du = cp.Variable((self.nu, self.N))

        # Absolute bounds for this subsystem
        u_min_abs = self.LBU[self.u_ids]
        u_max_abs = self.UBU[self.u_ids]
        x_min_abs = self.LBX[self.x_ids]
        x_max_abs = self.UBX[self.x_ids]

        # Add small margin to avoid numerical issues
        margin_u = 1e-3
        margin_x = 1e-3

        # Shifted bounds in delta coords around reference (slide 6) :contentReference[oaicite:1]{index=1}
        dx_min = (x_min_abs - self.xref_param) + margin_x
        dx_max = (x_max_abs - self.xref_param) - margin_x
        du_min = (u_min_abs - self.uref_param) + margin_u
        du_max = (u_max_abs - self.uref_param) - margin_u

        # Cost
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(dx[:, k], Q) + cp.quad_form(du[:, k], R)
        cost += cp.quad_form(dx[:, self.N], P)

        # Constraints
        cons = [dx[:, 0] == self.dx0_param]
        for k in range(self.N):
            cons += [dx[:, k+1] == self.A @ dx[:, k] + self.B @ du[:, k]]
            cons += [dx_min <= dx[:, k], dx[:, k] <= dx_max]
            cons += [du_min <= du[:, k], du[:, k] <= du_max]
        cons += [dx_min <= dx[:, self.N], dx[:, self.N] <= dx_max]

        # -------- Terminal set constraint as parameters --------
        self.max_F_rows = 250
        self.Ff_param = cp.Parameter((self.max_F_rows, self.nx))
        self.ff_param = cp.Parameter(self.max_F_rows)
        self.nF_param = cp.Parameter(nonneg=True, integer=True)

        # Safe default: no active terminal inequalities until first update
        self.Ff_param.value = np.zeros((self.max_F_rows, self.nx))
        self.ff_param.value = np.zeros(self.max_F_rows)
        self.nF_param.value = 0

        cons += [self.Ff_param[:self.nF_param, :] @ dx[:, self.N] <= self.ff_param[:self.nF_param]]

        # Store variables
        self.dx = dx
        self.du = du

        # Build the optimization problem ONCE
        self.ocp = cp.Problem(cp.Minimize(cost), cons)

        # cache
        self._last_xref = None
        self._last_uref = None

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

        # x0 is already reduced when called from the subsystem test,
        # OR it’s full state but sliced before calling in your notebook.

        # D3.2: x_target is scalar ref (vx, vy, vz, gamma)
        ref = 0.0 if x_target is None else float(x_target)

        # Compute reference steady state
        x_ref, u_ref = self.steady_state_from_ref(ref)

        self.xref_param.value = x_ref
        self.uref_param.value = u_ref
        self.dx0_param.value  = x0 - x_ref

        # Rebuild terminal set if ref changed
        if (self._last_xref is None or
            not np.allclose(x_ref, self._last_xref) or
            self._last_uref is None or
            not np.allclose(u_ref, self._last_uref)):

            Xf = self._build_terminal_set_for_ref(x_ref, u_ref)
            Ff, ff = Xf.A, Xf.b
            nF = Ff.shape[0]
            if nF > self.max_F_rows:
                raise RuntimeError(f"Xf too large: {nF} rows > {self.max_F_rows}")

            Fpad = np.zeros((self.max_F_rows, self.nx))
            fpad = np.zeros(self.max_F_rows)
            Fpad[:nF, :] = Ff
            fpad[:nF] = ff

            self.Ff_param.value = Fpad
            self.ff_param.value = fpad
            self.nF_param.value = nF

            self._last_xref = x_ref.copy()
            self._last_uref = u_ref.copy()

        # Solve
        self.ocp.solve(solver=cp.OSQP, warm_start=True,
                    max_iter=20000, eps_abs=1e-4, eps_rel=1e-4)

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"MPC failed: {self.ocp.status}")

        du0 = self.du[:, 0].value
        u0  = u_ref + du0

        x_traj = self.dx.value + x_ref[:, None]
        u_traj = self.du.value + u_ref[:, None]

        return u0, x_traj, u_traj

        # YOUR CODE HERE
        #################################################

    def steady_state_from_ref(self, ref):
        """
        Subclasses must implement:
        - ref is scalar: vx_ref, vy_ref, vz_ref, or gamma_ref [rad]
        - return x_ref (reduced), u_ref (reduced)
        """
        raise NotImplementedError

    def _build_terminal_set_for_ref(self, x_ref, u_ref):
        """
        Build terminal invariant set for given reference (x_ref, u_ref)
        """
        # Add small margin to avoid numerical issues
        margin_x = 1e-3
        margin_u = 1e-3


        # delta bounds around THIS reference
        dx_min = (self.LBX[self.x_ids] - x_ref) + margin_x
        dx_max = (self.UBX[self.x_ids] - x_ref) - margin_x
        du_min = (self.LBU[self.u_ids] - u_ref) + margin_u
        du_max = (self.UBU[self.u_ids] - u_ref) - margin_u

        Hx_list, hx_list = [], []

        for i in range(self.nx):
            if np.isfinite(dx_max[i]):
                e = np.zeros(self.nx); e[i] = 1.0
                Hx_list.append(e); hx_list.append(dx_max[i])
            if np.isfinite(dx_min[i]):
                e = np.zeros(self.nx); e[i] = -1.0
                Hx_list.append(e); hx_list.append(-dx_min[i])

        Hx = np.vstack(Hx_list)
        hx = np.array(hx_list)

        Hu = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        hu = np.hstack([du_max, -du_min])

        # enforce terminal feedback du = -K dx
        Hx_u = -Hu @ self.K
        hx_u = hu

        H0 = np.vstack([Hx, Hx_u])
        h0 = np.hstack([hx, hx_u])

        X0 = Polyhedron.from_Hrep(H0, h0)
        Acl = self.A - self.B @ self.K

        return self._max_invariant_set(Acl, X0)
    
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

    
