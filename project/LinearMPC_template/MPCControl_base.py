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
    Q: np.ndarray
    R: np.ndarray
    subsys_name: str

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
        UBX = np.array([50,  50,  50,
                        np.deg2rad(10),  np.deg2rad(10),  50,
                        50,  50,  50,
                        50,  50, 50])
        LBX = np.array([-50, -50, -50,
                        -np.deg2rad(10), -np.deg2rad(10), -50,
                        -50, -50, -50,
                        -50, -50, 0.0])
        self.UBU = UBU[self.u_ids]
        self.LBU = LBU[self.u_ids]
        self.UBX = UBX[self.x_ids]
        self.LBX = LBX[self.x_ids]
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

        # Cache for terminal set to avoid computing it if target same as for previous call of get_u
        self._cached_xt = None
        self._cached_ut = None
        self._cached_Xf = None

        self._setup_controller() # this method is called at initialization of the controller -> the optimization problem and its variables are all acessible

    def _setup_controller(self) -> None:

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_par = cp.Parameter((self.nx,), name='x0')
        self.xt_par = cp.Parameter((self.nx,), name='xtarget')
        self.ut_par = cp.Parameter((self.nu,), name= 'utarget')
        
        x_diff = self.x_var - cp.reshape(self.xt_par, (self.nx, 1))
        u_diff = self.u_var - cp.reshape(self.ut_par, (self.nu, 1))

        # Costs (objective function)
        self.cost = 0
        for k in range(self.N):
            self.cost += cp.quad_form(x_diff[:,k], self.Q)
            self.cost += cp.quad_form(u_diff[:,k], self.R)
        
        # Terminal cost
        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K
        self.cost += cp.quad_form(x_diff[:, -1], P)        

        # System (equality) constraint
        self.base_constraints = []
        self.base_constraints.append(self.x_var[:, 0] == self.x0_par)
        self.base_constraints.append(self.A @ x_diff[:, :-1] + self.B @ u_diff == x_diff[:, 1:]) # x^+ - xt = A(x-xt) + B(u-ut)

        # Inequality constraints
        self.base_constraints.append(self.X.A @ self.x_var <= self.X.b.reshape(-1, 1)) # x in X for all k = 0, ..., N

        self.base_constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1)) # u in U for all k = 0, ..., N-1

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, show_Xf: bool, x_target: np.ndarray = None, u_target: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
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

        # Build terminal set + terminal constraints USING (x_t, u_t) IF targets have changed sice last call of get_u:
        
        # If the invariant set has never been constructef yet, compute it
        if self._cached_xt is None or self._cached_ut is None or self._cached_Xf is None:
            self.X_f = self.compute_terminal_set_about_target(self.xt_par.value, self.ut_par.value, max_iter=50)
        else:
            # If x_target and u_target same as previous one, don't compute the invariant set again
            if np.allclose(self.xt_par.value, self._cached_xt, rtol=1e-10, atol=1e-10) and np.allclose(self.ut_par.value, self._cached_ut, rtol=1e-10, atol=1e-10):
                self.X_f = self._cached_Xf
            # If x_target or u_target changed with respect to their value the last time u_get was called, compute the invariant set again
            else:
                self.X_f = self.compute_terminal_set_about_target(self.xt_par.value, self.ut_par.value, max_iter=50)

        # If empty terminal set -> infeasible
        if self.X_f.Vrep.V.size == 0:
            raise RuntimeError(
                f"[{self.subsys_name}] Terminal set is EMPTY around current target; QP will be infeasible."
            )
        
        self._cached_xt = self.xt_par.value.copy()
        self._cached_ut = self.ut_par.value.copy()
        self._cached_Xf = self.X_f

        x_diff_N = self.x_var[:, -1] - self.xt_par  # (nx,)

        terminal_constraints = []
        terminal_constraints.append(self.X_f.A @ x_diff_N <= self.X_f.b) 
        terminal_constraints.append(self.U.A @ (self.ut_par + self.K @ x_diff_N) <= self.U.b)

        if show_Xf == True :
            self.plot_Xf(self.xt_par.value)

        # Build and solve FINAL OCP (base + terminal)
        ocp = cp.Problem(cp.Minimize(self.cost), self.base_constraints + terminal_constraints)

        ocp.solve(
            solver=cp.PIQP,
            warm_start=True,
            max_iter=20000,
            eps_abs=1e-4,
            eps_rel=1e-4
        )

        if ocp.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP problem failed: {ocp.status}")

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        # print(f"u0 for system {self.subsys_name} is {u0}\n")
        # print(f"x_ol for system {self.subsys_name} is {x_traj}\n")
        # print(f"u_ol for system {self.subsys_name} is {u_traj}\n")

        return u0, x_traj, u_traj

    def compute_terminal_set_about_target(
        self,
        xt: np.ndarray,
        ut: np.ndarray,
        max_iter: int = 50,
    ) -> Polyhedron:
        """
        Compute maximal invariant set for delta dynamics:
            x+ - xt = (A + B K) (x - xt)
        subject to:
            x - xt in X_delta = {x - xt : x in X}
            K (x - xt) in U_delta = {u - ut : u in U}
        """
        # Build delta-sets wrt CURRENT reference (x_t, u_t)
        # X_delta = {x-xt | x in X} -> bounds shift
        X_delta = Polyhedron.from_bounds(self.LBX - xt, self.UBX - xt)
        U_delta = Polyhedron.from_bounds(self.LBU - ut, self.UBU - ut)

        A_cl = self.A + self.B @ self.K

        KU_delta = Polyhedron.from_Hrep(U_delta.A @ self.K, U_delta.b)
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

        return Omega


    def plot_Xf(self, x_target):
        pairs = list(combinations(range(self.nx), 2))

        if x_target is not None:
            Xf_shifted = Polyhedron.from_Hrep(self.X_f.A, self.X_f.b + self.X_f.A @ x_target)
        else:
            Xf_shifted = self.X_f

        if Xf_shifted.Vrep.V.size == 0:
            print('X_f is empty, skipping plot')
            return

        if len(pairs) == 0:
            print(f"System {self.subsys_name} is 1st order")

            xmin = Xf_shifted.Vrep.V[0, 0]
            xmax = Xf_shifted.Vrep.V[1, 0]

            plt.figure()
            plt.hlines(0, xmin, xmax, color='y', linewidth=6)
            plt.plot([xmin, xmax], [0, 0], "o", color='y')
            plt.yticks([])
            plt.xlabel(self.red_state_names[0])
            plt.title(rf"$X = [{xmin:.4g},\, {xmax:.4g}]$")
            plt.grid(True, axis="x")
            plt.show()

        else:
            print(f"System {self.subsys_name} is high order (>1)")
            for (i, j) in pairs:
                fig, ax = plt.subplots(1, 1)

                # plot feasible set and terminal set in the same projection
                Xf_shifted.projection(dims=(i, j)).plot(ax, color='y', opacity=0.35, label=r'$\mathcal{X}_f$')

                ax.set_xlabel(self.red_state_names[i])
                ax.set_ylabel(self.red_state_names[j])
                ax.set_title(f"Projection: {self.red_state_names[i]} vs {self.red_state_names[j]}")
                ax.grid(True)

                fig.suptitle(f"Terminal set of subsystem {self.subsys_name} w.r.t. states {self.red_state_names[i]} and {self.red_state_names[j]}")
                plt.legend()
                plt.show()


