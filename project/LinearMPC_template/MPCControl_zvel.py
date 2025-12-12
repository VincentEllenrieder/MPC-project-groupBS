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
        Q = np.diag([20.0])
        R = np.diag([2.0])

        # 2. LQR
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Absolute limits: [40, 80]
        # Delta limits: [40 - us, 80 - us]
        u_min = np.array([40.0]) - self.us
        u_max = np.array([80.0]) - self.us

        # State Limits
        # Physically v_z is unbounded, but we need bounds for the invariant set.
        # Let's say +/- 15 m/s is a reasonable safe envelope.
        x_max = np.array([15.0])
        x_min = np.array([-15.0])

        # 4. Terminal Set
        A_cl = self.A - self.B @ K

        M = np.vstack([np.eye(self.nx), -np.eye(self.nx), -K, K])
        m = np.hstack([x_max, -x_min, u_max, -u_min])
        
        m = m.reshape(-1)
        #X_poly = Polyhedron(A=M, b=m)
        X_poly = Polyhedron.from_Hrep(M, m)
        term_set = self.max_invariant_set(A_cl, X_poly)

        # 5. Build
        self._build_problem(Q, R, P, u_min, u_max, x_min, x_max, term_set)
        

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