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
        # 1. Tuning Weights
        # State: [w_y, beta, v_x]
        # Prioritize velocity tracking (v_x) and stability (beta)
        Q = np.diag([1.0, 10.0, 10.0])
        R = np.diag([10.0])

        # 2. Terminal Cost (DLQR)
        # Calculates the infinite horizon cost for stability
        K, P, _ = dlqr(self.A, self.B, Q, R)

        # 3. Constraints (Relative to Trim)
        # Input Constraints: Servo angle +/- 15 deg (0.26 rad)
        # u_min - u_s <= delta_u <= u_max - u_s
        u_lim = 0.2618 # 15 degrees in rad
        u_min = -u_lim - self.us
        u_max =  u_lim - self.us

        # State Constraints: Beta +/- 10 deg (0.1745 rad)
        # States: [w_y, beta, v_x]
        # We need generic bounds for all 3 states for the polytope calculation.
        inf = 10.0 # Large finite number for "unbounded" states
        x_lim_beta = 0.1745
        
        # Order: [w_y, beta, v_x]
        x_max = np.array([inf, x_lim_beta, inf])
        x_min = -x_max

        # 4. Terminal Set Calculation (Invariant Set)
        # Calculate the closed-loop dynamics: A_cl = A - B*K
        A_cl = self.A - self.B @ K

        # Define the constraint polytope X: {x | Mx <= m}
        # This includes state constraints (x_min <= x <= x_max) 
        # AND input constraints mapped to state (-u_min <= Kx <= u_max)
        # Note: u = -Kx, so u_min <= -Kx <= u_max  =>  Kx <= -u_min  AND -Kx <= u_max
        
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

        try:
            # Create the initial Polytope
            X_poly = Polyhedron.from_Hrep(M, m)
            
            # Compute the Maximal Invariant Set
            term_set = self.max_invariant_set(A_cl, X_poly)
        except Exception as e:
            print(f"Warning: Terminal set calculation failed: {e}")
            print("Proceeding with Terminal Cost P only (Stable but not strictly recursive).")
            term_set = None
        '''
        # Create the initial Polytope
        #X_poly = Polyhedron(A=M, b=m)
        X_poly = Polyhedron.from_Hrep(M, m)

        # Compute the Maximal Invariant Set using your manual function
        term_set = self.max_invariant_set(A_cl, X_poly)
        '''

        # 5. Build the Problem
        # Pass everything to the base class helper to construct the CVXPY problem
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
        
        #return super().get_u(x0, x_target, u_target)
    '''