import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations
from control import dlqr, place
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base

class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    subsys_name: str = 'z'

    # Q, R matrices for cost function
    Q = np.diag(np.array([0.05, 50]))
    R = np.array([[0.01]])

    # Noise model
    w_min = -15
    w_max = 5
    W = Polyhedron.from_bounds(w_min, w_max)

    def _setup_controller(self) -> None:

        self.z_var = cp.Variable((self.nx, self.N+1), name='z')
        self.v_var = cp.Variable((self.nu, self.N), name='v')
        self.x0_par = cp.Parameter((self.nx,), name='x0')

        z_diff = self.z_var - cp.reshape(self.xs, (self.nx, 1))
        v_diff = self.v_var - cp.reshape(self.us, (self.nu, 1))

        # Costs (objective function)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(z_diff[:,k], self.Q)
            cost += cp.quad_form(v_diff[:,k], self.R)
        
        # Terminal cost
        K_lqr, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        cost += cp.quad_form(z_diff[:, -1], P)

        # Sets definitions
        self.K = -place(self.A, self.B, [0.6, 0.59])
        self.min_inv_set_stop_condition = 6*1e-2
        print('Min invariant set stopping criterion :', self.min_inv_set_stop_condition)
        print('A =', self.A)
        print('B =', self.B)
        print('K =', self.K)
        A_cl = self.A + self.B @ self.K 
        print('Closed-loop control eigenvalues:', np.linalg.eig(A_cl).eigenvalues)
    
        self.E = self.min_robust_invariant_set(A_cl, self.W.affine_map(self.B)) # A_cl is nx x nx but W is a 1D set, as given in project description, we use B @ w to map w from 1D to the state dimension nx = 2
        KE = self.E.affine_map(self.K)  
        X_tilde = self.X - self.E
        self.U_tilde = self.U - KE
        print('U_tilde bounds :', self.U_tilde.b)

        # Equality constraints
        constraints = []
        constraints.append(self.A @ z_diff[:, :-1] + self.B @ v_diff == z_diff[:, 1:]) # z^+ - xs = A(z-xs) + B(v-us)

        # Inequality constraints
        constraints.append(self.E.A @ (self.x0_par - self.z_var[:, 0]) <= self.E.b)                # x0 - z0 in E
        constraints.append(X_tilde.A @ self.z_var[:, :-1] <= X_tilde.b.reshape(-1, 1))   # z in X_tilde for all k = 1,..., N-1
        constraints.append(self.U_tilde.A @ self.v_var <= self.U_tilde.b.reshape(-1, 1))           # v in U_tilde for all k = 1,..., N-1

        # Terminal constraint 
        K_f = -place(self.A, self.B, [0.75, 0.8])
        self.X_tilde_delta = Polyhedron.from_Hrep(X_tilde.A, X_tilde.b - (X_tilde.A @ self.xs.reshape(-1, 1)).reshape(-1)) # shifted X_tilde around xs
        self.U_tilde_delta = Polyhedron.from_Hrep(self.U_tilde.A, self.U_tilde.b - (self.U_tilde.A @ self.us.reshape(-1, 1)).reshape(-1)) # shifted U_tilde around us
        self.KU_tilde_delta = Polyhedron.from_Hrep(self.U_tilde_delta.A @ K_f, self.U_tilde_delta.b)
        self.X_int_KU_tilde_delta = self.X_tilde_delta.intersect(self.KU_tilde_delta)

        A_cl_f = self.A + self.B @ K_f

        self.X_f_tilde_delta = self.max_invariant_set(A_cl_f, self.X_int_KU_tilde_delta, max_iter=50)

        if self.X_f_tilde_delta.Vrep.V.size == 0:
            print('Terminal set is EMPTY around current target -> QP will be infeasible')

        constraints.append(self.X_f_tilde_delta.A @ z_diff[:, -1] <= self.X_f_tilde_delta.b)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.x0_par.value = x0

        self.ocp.solve()

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP problem failed: {self.ocp.status}")
        
        z0 = self.z_var.value[:, 0]
        v0 = self.v_var.value[:, 0]

        u0 = v0 + self.K @ (x0 - z0)
        u0 = np.clip(u0, self.LBU, self.UBU) # clipping for numerical inconsistencies that make u0 drop below 40

        # x_traj and u_traj needed as output: set to 0 and computed at a later stage
        x_traj = np.zeros((self.nx, self.N+1))
        u_traj = np.zeros((self.nu, self.N))

        return u0, x_traj, u_traj
    
    def plot_invariant_set_evolution(self, plot_dims=(0, 1)):
        fig, ax = plt.subplots(1, 1)
        ax.set_title(r"Minimal robust invariant set $\mathcal{E}$ evolution")

        Vlast = self.Omega_hist[-1].Vrep.V
        if Vlast is None or Vlast.size == 0:
            ax.text(0.02, 0.95, "Vrep empty: nothing to plot", transform=ax.transAxes)
            ax.grid(True)
            plt.show()
            return

        dim = Vlast.shape[1]

        # --- OPACITY: early iterations DARK, later iterations LIGHT ---
        alpha_max = 0.80   # iteration 1 (very dark)
        alpha_mid = 0.12   # after the "knee" (already quite light)
        alpha_min = 0.02   # last iterations
        knee_frac = 0.25   # first 25% iterations fade fast

        n = len(self.Omega_hist)
        knee = max(1, int(np.ceil(knee_frac * (n - 1))))

        def alpha_for(k: int) -> float:
            # k in [0, n-1]
            if n <= 1:
                return alpha_max

            if k <= knee:
                # fast fade: alpha_max -> alpha_mid
                t = k / knee if knee > 0 else 1.0
                return alpha_max + (alpha_mid - alpha_max) * t
            else:
                # slow fade: alpha_mid -> alpha_min
                t = (k - knee) / (n - 1 - knee)
                return alpha_mid + (alpha_min - alpha_mid) * t

        if dim == 1:
            ax.set_yticks([])
            ax.set_xlabel(self.red_state_names[plot_dims[0]])
            ax.grid(True, axis="x")

            for k, Ek in enumerate(self.Omega_hist):
                Vk = Ek.Vrep.V
                if Vk is None or Vk.size == 0:
                    continue
                xmin = float(np.min(Vk[:, 0]))
                xmax = float(np.max(Vk[:, 0]))
                ax.hlines(
                    0.0, xmin, xmax,
                    linewidth=6,
                    alpha=alpha_for(k),
                )

            # Highlight FINAL set clearly
            Vk = self.Omega_hist[-1].Vrep.V
            xmin = float(np.min(Vk[:, 0]))
            xmax = float(np.max(Vk[:, 0]))
            ax.hlines(0.0, xmin, xmax, linewidth=7, alpha=0.95)
            ax.set_title(rf"$\mathcal{{E}}$ evolution (final = [{xmin:.4g}, {xmax:.4g}])")

        else:
            dims = plot_dims if plot_dims is not None else (0, 1)
            ax.set_xlabel(self.red_state_names[dims[0]])
            ax.set_ylabel(self.red_state_names[dims[1]])
            ax.grid(True)

            for k, Ek in enumerate(self.Omega_hist):
                try:
                    Ek.projection(dims=dims).plot(
                        ax,
                        opacity=alpha_for(k),
                        show_edges=True,
                        edge_width=1.0,
                    )
                except Exception:
                    pass

            # Highlight FINAL set clearly
            try:
                self.Omega_hist[-1].projection(dims=dims).plot(
                    ax,
                    opacity=0.25,
                    show_edges=True,
                    edge_width=2.0,
                )
            except Exception:
                pass

        plt.show()


    def min_robust_invariant_set(
        self,
        A_cl: np.ndarray,
        W: Polyhedron,
        max_iter: int = 50,
        plot_evolution: bool = True,
        plot_dims=(0, 1),
    ) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        self.Omega_hist = []
        A_cl_ith_power = np.eye(nx)

        i = 1
        while i <= max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, i)
            Omega_next = Omega + W.affine_map(A_cl_ith_power)
            Omega_next.minHrep()
            _ = Omega_next.Vrep  # force V-rep computation
            self.Omega_hist.append(Omega_next)
            norm = np.linalg.matrix_norm(A_cl_ith_power, ord=2)
            print(f'||A+BK||^{i} =', norm)
            if norm < self.min_inv_set_stop_condition:
                Omega = Omega_next
                print(f"Minimum robust invariant set found after {i} iterations!\n")
                break

            if i == max_iter:
                print('Minimal robust invariant set computation did NOT converge after {0} iterations.'.format(i))
            else:
                print("Not yet convgerged at iteration {0}" .format(i))

            Omega = Omega_next
            i += 1

        if plot_evolution and len(self.Omega_hist) > 0:
            self.plot_invariant_set_evolution(plot_dims=plot_dims)

        return Omega
    
    def max_invariant_set(self, A_cl: np.ndarray, X_int_KU: Polyhedron, max_iter: int = 50) -> Polyhedron:

        Omega = X_int_KU
        i = 0
        while i < max_iter:
            H, h = Omega.A, Omega.b
            pre_omega = Polyhedron.from_Hrep(H @ A_cl, h)
            Omega_new = pre_omega.intersect(Omega)
            Omega_new.minHrep(True)
            _ = Omega_new.Vrep

            if Omega == Omega_new:
                Omega = Omega_new
                print(f"Maximum invariant set found after {i+1} iterations!\n")
                break

            print(f"Not yet converged at iteration {i+1}")
            Omega = Omega_new
            i += 1

        return Omega

    
    def plot_set(self, set: Polyhedron, set_name: str):
        pairs = list(combinations(range(set.Vrep.V.shape[1]), 2))

        if set.Vrep.V.size == 0:
            print(f'{set_name} is empty, skipping plot')
            return

        if len(pairs) == 0:
            print(f"{set_name} is of dimension 1")

            xmin = set.Vrep.V[0, 0]
            xmax = set.Vrep.V[1, 0]

            plt.figure()
            plt.hlines(0, xmin, xmax, color='#1f2eb4', linewidth=6)
            plt.plot([xmin, xmax], [0, 0], "o", color="#1f2eb4")
            plt.yticks([])
            plt.xlabel(self.red_state_names[0])
            plt.title(rf"${set_name} = [{xmin:.4g},\, {xmax:.4g}]$")
            plt.grid(True, axis="x")
            plt.show()

        else:
            print(f"{set_name} is of dimension {set.Vrep.V.shape[1]}")
            for (i, j) in pairs:
                fig, ax = plt.subplots(1, 1)

                if set_name == 'Terminal set':
                    set_label = '\mathcal{X}_f'
                elif set_name == 'Error set':
                    set_label = '\mathcal{E}'
                else:
                    set_label = None
                set.projection(dims=(i, j)).plot(ax, color='b', opacity=0.35, label=fr'${set_label}$')

                ax.set_xlabel(self.red_state_names[i])
                ax.set_ylabel(self.red_state_names[j])
                ax.grid(True)

                fig.suptitle(f"{set_name} of subsystem {self.subsys_name} w.r.t. states {self.red_state_names[i]} and {self.red_state_names[j]}")
                plt.legend()
                plt.show()

