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
    subsys_name: str
    UBX: np.ndarray
    LBX: np.ndarray
    UBU: np.ndarray
    LBU: np.ndarray
    X: Polyhedron
    U: Polyhedron


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

        self.UBU = np.array([np.deg2rad(15), np.deg2rad(15), 80, 20])
        self.LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])
        self.UBX = np.array([np.inf,  np.inf,  np.inf,
                        np.deg2rad(10),  np.deg2rad(10),  np.inf,
                        np.inf,  np.inf,  np.inf,
                        np.inf,  np.inf, np.inf])
        self.LBX = np.array([-np.inf, -np.inf, -np.inf,
                        -np.deg2rad(10), -np.deg2rad(10), -np.inf,
                        -np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, 0.0])
        self.UBU = self.UBU[self.u_ids]
        self.LBU = self.LBU[self.u_ids]
        self.UBX = self.UBX[self.x_ids]
        self.LBX = self.LBX[self.x_ids]
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

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        self.ocp = ...

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

        u0 = ...
        x_traj = ...
        u_traj = ...

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
