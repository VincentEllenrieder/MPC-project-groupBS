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

        self._setup_controller() # this method is called at initialization of the controller -> the optimization problem and its variables are all acessible

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

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
        constraints.append(self.A @ x_diff[:, :-1] + self.B @ u_diff == x_diff[:, 1:]) # x^+ - xt = A(x-xt) + B(u-ut)

        # Inequality constraints
        constraints.append(self.X.A @ self.x_var <= self.X.b.reshape(-1, 1)) # x in X for all k = 0, ..., N

        constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1)) # u in U for all k = 0, ..., N-1

        # Terminal constraints
        U_Delta = Polyhedron.from_bounds(self.LBU - self.us, self.UBU - self.us)
        X_Delta = Polyhedron.from_bounds(self.LBX - self.xs, self.UBX - self.xs)

        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        A_cl = self.A + self.B @ K

        KU_Delta = Polyhedron.from_Hrep(U_Delta.A @ K, U_Delta.b)  # KU_Delta = {x - xs : K * (x - xs) in U_Delta}
        X_int_KU = X_Delta.intersect(KU_Delta)

        i = 0
        max_iter = 50
        Omega = X_int_KU
        while i < max_iter :
            H, h = Omega.A, Omega.b
            pre_omega = Polyhedron.from_Hrep(H @ A_cl, h)
            Omega_new = pre_omega.intersect(Omega)
            Omega_new.minHrep(True)
            _ = Omega_new.Vrep # TODO: this is a tempary fix since the contains() method is not robust enough when both inner and outer polyhera only has H-rep (from solution of ex 4)
            if Omega == Omega_new :
                Omega = Omega_new
                print("Maximum invariant set found after {0} iterations !\n" .format(i+1))
                break
            print("Not yet convgerged at iteration {0}" .format(i+1))
            Omega = Omega_new
            i += 1

        self.X_f = Omega

        constraints.append(self.X_f.A @ x_diff[:, -1] <= self.X_f.b.reshape(-1, 1)) #x_N - xt in X_f and x_N in X

        # If tracking problem done in "true" values (not delta formulation), then need to add the following constraint (previously in X_f when tracking not present).
        # Could remove this constraint (that corresponds to u-us = K(x - xs) in U) if X_f was recomputed with the trackpoints, but this is not handy as we would
        # need to know the value of x_target by passing it as an attribute to the class (here just a CVXPY parameter, not an attribute, that is still unitialized at the moment 
        # of _setup_controller() call, and x_target not yet know. To change this, it would need quite some rewriting of MPCVelControl and the way mpc object is initialized).
        # So, to keep the usage of the CVXPY parameter, we need to use all sets X, U and X_f as the ones computed during regulation, but this means that fot the final
        # constraint, we need to add this last constraint as X_f was computed when tracking was not present. See goodnotes for more details.

        constraints.append(self.U.A @ (self.ut_par + K @ x_diff[:, -1]) <= self.U.b.reshape(-1, 1)) # u_N = ut + K(x_N - xt) in U

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
        self, x0: np.ndarray, show_Xf: bool, x_target: np.ndarray = None, u_target: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        
        self.x0_par.value = x0

        if x_target is None:
            self.xt_par.value = x_target
        else:
            self.xt_par.value = self.xs

        if u_target is None:
            self.ut_par.value = self.us
        else:
            self.ut_par.value = u_target

        if show_Xf == True :
            self.plot_Xf(x_target)

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
        print(f"u0 for system {self.subsys_name} is {u0}\n")
        print(f"x_ol for system {self.subsys_name} is {x_traj}\n")
        print(f"u_ol for system {self.subsys_name} is {u_traj}\n")

        return u0, x_traj, u_traj
            

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

            
        # YOUR CODE HERE
        #################################################

