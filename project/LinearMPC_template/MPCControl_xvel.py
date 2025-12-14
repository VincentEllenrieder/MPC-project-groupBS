import numpy as np
from .MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron


class MPCControl_xvel(MPCControl_base):
    # Indices from sys.x: [w_y (1), beta (4), v_x (6), x (9)]
    # We exclude x(9) for velocity control as per instructions
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1]) # Input is delta_2 (servo 2)

    def _setup_controller(self) -> None:
        # 1. Tuning
        # Prioritize velocity tracking (v_x) and stability (beta)
        Q = np.diag([1.0, 1.0, 1.0])
        R = np.diag([80.0])

        # 2. Terminal Cost (DLQR Infinite horizon cost)
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Input Constraints: Servo angle +/- 15 deg (0.26 rad)
        u_lim = 0.2618
        u_min = -u_lim - self.us
        u_max =  u_lim - self.us

        # State Constraints: Beta +/- 10 deg (0.1745 rad)
        inf = 10.0 #(for "unbounded" states)
        x_lim_beta = 0.1745
        x_max = np.array([inf, x_lim_beta, inf])
        x_min = -x_max

        try:

            # 4. Terminal Set Calculation (Invariant Set)
            A_cl = self.A - self.B @ K
            
            M = np.vstack([
                np.eye(self.nx),   # I*x <= x_max
                -np.eye(self.nx),  # -I*x <= -x_min
                -K,                # -K*x <= u_max
                K                  # K*x <= -u_min
            ])
            
            m = np.hstack([
                x_max,
                -x_min,
                u_max,
                -u_min
            ])

            m = m.reshape(-1)

            X_poly = Polyhedron.from_Hrep(M, m)
            self.term_set = self.max_invariant_set(A_cl, X_poly)

        except Exception as e:
            print(f"Warning: Invariant Set calculation failed for {self.__class__.__name__}")
            print(f"Error details: {e}")
            print("Proceeding with Terminal Cost P only (Standard Stability).")
            self.term_set = None

        # 5. Build the Problem
        self._build_problem(Q, R, P, u_min, u_max, x_min, x_max)