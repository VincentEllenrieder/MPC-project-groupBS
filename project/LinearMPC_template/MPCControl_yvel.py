import numpy as np

from .MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron


class MPCControl_yvel(MPCControl_base):
    # Indices from sys.y: [w_x(0), alpha(3), v_y(7)]
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # 1. Tuning
        # Penalize velocity (v_y) and angle (alpha)
        Q = np.diag([1.0, 10.0, 10.0])
        R = np.diag([10.0])

        # 2. LQR for Terminal Cost/Controller
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Input: +/- 15 deg (0.2618 rad)
        u_lim = 0.2618
        u_min = -u_lim - self.us
        u_max =  u_lim - self.us

        # State: Alpha (Index 1) +/- 10 deg (0.1745 rad)
        inf = 10.0
        x_lim_alpha = 0.1745
        x_max = np.array([inf, x_lim_alpha, inf])
        x_min = -x_max

        # 4. Terminal Set Calculation
        # A_cl = A - B*K
        A_cl = self.A - self.B @ K

        # Define the constraint polytope {x | Mx <= m}
        # Includes: x_min <= x <= x_max  AND  u_min <= -Kx <= u_max
        M = np.vstack([np.eye(self.nx), -np.eye(self.nx), -K, K])
        m = np.hstack([x_max, -x_min, u_max, -u_min])
        
        m = m.reshape(-1)
        #X_poly = Polyhedron(A=M, b=m)
        X_poly = Polyhedron.from_Hrep(M, m)
        
        # Use your manual function
        term_set = self.max_invariant_set(A_cl, X_poly)

        # 5. Build
        self._build_problem(Q, R, P, u_min, u_max, x_min, x_max, term_set)
        #################################################





    '''
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
    '''