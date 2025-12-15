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

        self.Delta_x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.Delta_u_var = cp.Variable((self.nu, self.N), name='u')
        self.Delta_x0 = cp.Parameter((self.nx,), name='x0')
        # self.x_target = cp.Parameter((self.nx,), name='xt')
        # self.u_target = cp.Parameter((self.nu,), name='ut')

        # Costs (objective function)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.Delta_x_var[:,k], self.Q)
            cost += cp.quad_form(self.Delta_u_var[:,k], self.R)
        
        # Terminal cost
        K, P, _ = dlqr(self.A, self.B, self.Q, self.R)
        K = -K
        cost += cp.quad_form(self.Delta_x_var[:, -1], P)        

        # System (equality) constraint
        constraints = []
        constraints.append(self.Delta_x_var[:, 0] == self.Delta_x0)
        constraints.append(self.A @ self.Delta_x_var[:, :-1] + self.B @ self.Delta_u_var == self.Delta_x_var[:, 1:])

        # Inequality constraints

        constraints.append(
            self.X.A @ self.Delta_x_var[:, :-1]
            <= self.X.b.reshape(-1, 1) - (self.X.A @ self.xs).reshape(-1, 1) #- self.X.A @ self.x_target.reshape(-1, 1)
        )

        constraints.append(
            self.U.A @ self.Delta_u_var
            <= self.U.b.reshape(-1, 1) - (self.U.A @ self.us).reshape(-1, 1) #- self.U.A @ self.u_target.reshape(-1, 1)
        )

        # Terminal constraint
        U_Delta = Polyhedron.from_bounds(
            self.LBU - self.us, # - self.u_target, 
            self.UBU - self.us, # - self.u_target
            )
        X_Delta = Polyhedron.from_bounds(
            self.LBX - self.xs, # - self.x_target,
            self.UBX - self.xs # - self.x_target
            )
        A_cl = self.A + self.B @ K
        KU_Delta = Polyhedron.from_Hrep(U_Delta.A @ K, U_Delta.b)
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

        constraints.append(self.X_f.A @ self.Delta_x_var[:, -1] <= self.X_f.b.reshape(-1, 1))

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

        # if x_target is None:
        #     x_target = np.zeros(self.nx)
        # if u_target is None:
        #     u_target = np.zeros(self.nu)
        # self.x_target.value = x_target
        # self.u_target.value = u_target
        
        self.Delta_x0.value = x0 - self.xs # - x_target

        self.ocp.solve(
            solver=cp.PIQP,
            warm_start=True,
            max_iter=20000,
            eps_abs=1e-4,
            eps_rel=1e-4
        )
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"MPC QP infeasible or failed, status={self.ocp.status}")

        u0 = self.Delta_u_var.value[:, 0] + self.us # + u_target
        x_traj = self.Delta_x_var.value + self.xs.reshape(-1, 1) # + x_target
        u_traj = self.Delta_u_var.value + self.us.reshape(-1, 1) # + u_target

        # plotting the terminal sets
        if show_Xf == True:
            
            pairs = list(combinations(range(self.nx), 2))

            if len(pairs) == 0:
                print(f"System {self.subsys_name} is 1st order")
                A = np.asarray(self.X_f.A).reshape(-1, 1)   # (m,1)
                b = np.asarray(self.X_f.b).reshape(-1)      # (m,)

                x = cp.Variable(1)

                constraints = [A @ x <= b]

                # xmin
                prob_min = cp.Problem(cp.Minimize(x[0]), constraints)
                prob_min.solve(solver=cp.ECOS)
                if prob_min.status not in ("optimal", "optimal_inaccurate"):
                    raise RuntimeError(f"Min problem failed: {prob_min.status}")
                xmin = float(x.value[0])

                # xmax
                prob_max = cp.Problem(cp.Maximize(x[0]), constraints)
                prob_max.solve(solver=cp.ECOS)
                if prob_max.status not in ("optimal", "optimal_inaccurate"):
                    raise RuntimeError(f"Max problem failed: {prob_max.status}")
                xmax = float(x.value[0])

                plt.figure()
                plt.hlines(0, xmin + self.xs, xmax + self.xs, color='y', linewidth=6)
                plt.plot([xmin + self.xs, xmax + self.xs], [0, 0], "o", color='y')
                plt.yticks([])
                plt.xlabel(rf"{self.red_state_names[0]}")
                plt.title(rf"$X_f = [{xmin:.4g},\, {xmax:.4g}]$")
                plt.grid(True, axis="x")
                plt.show()


            else:
                print(f"System {self.subsys_name} is high order (>1)")
                for (i, j) in pairs:
                    fig, ax = plt.subplots(1, 1)

                    # plot feasible set and terminal set in the same projection
                    self.X_f.projection(dims=(i, j)).plot(ax, color='y', opacity=0.35, label=r'$\mathcal{X}_f$')

                    ax.set_xlabel(self.red_state_names[i])
                    ax.set_ylabel(self.red_state_names[j])
                    ax.set_title(f"Projection: {self.red_state_names[i]} vs {self.red_state_names[j]}")
                    ax.grid(True)

                    fig.suptitle(f"Terminal set of subsystem {self.subsys_name} w.r.t. states {self.red_state_names[i]} and {self.red_state_names[j]}")
                    plt.legend()
                    plt.show()

            
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
