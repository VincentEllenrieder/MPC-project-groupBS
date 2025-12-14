import numpy as np

from .MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron


class MPCControl_roll(MPCControl_base):
    # Indices from sys.roll: [w_z(2), gamma(5)]
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self):
        # 1. Tuning
        # Penalize angle error (gamma) more than rate (w_z)
        Q = np.diag([1.0, 10.0])
        R = np.diag([10.0])

        # 2. Terminal Cost (DLQR Infinite horizon cost)
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Input: +/- 20
        u_lim = 20.0
        u_min = -u_lim - self.us
        u_max =  u_lim - self.us

        # State Limits
        inf = 10.0
        x_lim_gamma = 1.0 # +/- 60 deg (~1.0 rad)
        x_max = np.array([inf, x_lim_gamma])
        x_min = -x_max

        try:
            # 4. Terminal Set
            A_cl = self.A - self.B @ K

            M = np.vstack([np.eye(self.nx), -np.eye(self.nx), -K, K])
            m = np.hstack([x_max, -x_min, u_max, -u_min])
            m = m.reshape(-1)

            X_poly = Polyhedron.from_Hrep(M, m)
            self.term_set = self.max_invariant_set(A_cl, X_poly)

        except Exception as e:
            print(f"Warning: Invariant Set calculation failed for {self.__class__.__name__}")
            print(f"Error details: {e}")
            print("Proceeding with Terminal Cost P only (Standard Stability).")
            self.term_set = None

        # 5. Build
        self._build_problem(Q, R, P, u_min, u_max, x_min, x_max)