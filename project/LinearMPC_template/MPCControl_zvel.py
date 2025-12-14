import numpy as np

from .MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron


class MPCControl_zvel(MPCControl_base):
    # Indices from sys.z: [v_z(8)]
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self):
        # 1. Tuning
        # Heavily penalize velocity error
        Q = np.diag([10.0])
        R = np.diag([10.0])

        # 2. Terminal Cost (DLQR Infinite horizon cost)
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Absolute limits: [40, 80]
        # Delta limits: [40 - us, 80 - us]
        u_min = np.array([40.0]) - self.us
        u_max = np.array([80.0]) - self.us

        # State Limits
        # (Physically v_z is unbounded, but we need bounds for the invariant set.)
        inf = 10.0
        x_max = np.array([inf])
        x_min = np.array([-inf])

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
        