import numpy as np

from src.rocket import Rocket

from MPCControl_xvel_D3_1 import MPCControl_xvel_tuned_final
from MPCControl_yvel_D3_1 import MPCControl_yvel_tuned_final
from MPCControl_zvel_D3_1 import MPCControl_zvel_tuned_final
from MPCControl_roll_D3_1 import MPCControl_roll_tuned_final


class MPCVelControl:
    mpc_x: MPCControl_xvel_tuned_final
    mpc_y: MPCControl_yvel_tuned_final
    mpc_z: MPCControl_zvel_tuned_final
    mpc_roll: MPCControl_roll_tuned_final

    def __init__(self) -> None:
        pass

    def new_controller(self, rocket: Rocket, Ts: float, H: float) -> None:
        self.xs, self.us = rocket.trim()
        A, B = rocket.linearize(self.xs, self.us)

        self.mpc_x = MPCControl_xvel_tuned_final(A, B, self.xs, self.us, Ts, H)
        self.mpc_y = MPCControl_yvel_tuned_final(A, B, self.xs, self.us, Ts, H)
        self.mpc_z = MPCControl_zvel_tuned_final(A, B, self.xs, self.us, Ts, H)
        self.mpc_roll = MPCControl_roll_tuned_final(A, B, self.xs, self.us, Ts, H)

        return self

    def load_controllers(
        self,

        mpc_x: MPCControl_xvel_tuned_final,
        mpc_y: MPCControl_yvel_tuned_final,
        mpc_z: MPCControl_zvel_tuned_final,
        mpc_roll: MPCControl_roll_tuned_final,
    ) -> None:
        self.mpc_x = mpc_x
        self.mpc_y = mpc_y
        self.mpc_z = mpc_z
        self.mpc_roll = mpc_roll

        return self

    def estimate_parameters(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        return

    def get_u(
        self,
        t0: float,
        x0: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u0 = np.zeros(4)
        t_traj = np.arange(self.mpc_x.N + 1) * self.mpc_x.Ts + t0
        x_traj = np.zeros((12, self.mpc_x.N + 1))
        u_traj = np.zeros((4, self.mpc_x.N))

        if x_target is None: # if no tracking target given, do regulation (stabilize at the origin, as x_s = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            x_target = self.xs

        if u_target is None: # if no tracking given, the final control input in the closed-loop trajectory should be u_s = 0, 0, 66.66666667, 0
            u_target = self.us

        # these next 4 blocks of code use the methods get_u incorporated in each subsystem x, y, z and roll for finding :
        # 1. the first optimal control input (u0) for this subsystem among the complete control input vector of size n_u_subsys.
        # 2. the total open-loop trajectory of length N+1 (from 0 to N) of the states related to this subsystem among the complete set of states of size n_x_subsys x N+1.
        # 3. the total open-loop trajectory of length N (from 1 to N) of the the optimal control inputs related to this subsystem among the complete control input vector of size n_u_subsys x N+1
        # All these are compiled in the complete arrays of size | u0: 4 x 1 | x_traj: 12 x N+1 | u_traj: 4 x N
        u0[self.mpc_x.u_ids], x_traj[self.mpc_x.x_ids], u_traj[self.mpc_x.u_ids] = (
            self.mpc_x.get_u(
                x0[self.mpc_x.x_ids],
                x_target[self.mpc_x.x_ids],
                u_target[self.mpc_x.u_ids],
            )
        )
        u0[self.mpc_y.u_ids], x_traj[self.mpc_y.x_ids], u_traj[self.mpc_y.u_ids] = (
            self.mpc_y.get_u(
                x0[self.mpc_y.x_ids],
                x_target[self.mpc_y.x_ids],
                u_target[self.mpc_y.u_ids],
            )
        )
        u0[self.mpc_z.u_ids], x_traj[self.mpc_z.x_ids], u_traj[self.mpc_z.u_ids] = (
            self.mpc_z.get_u(
                x0[self.mpc_z.x_ids],
                x_target[self.mpc_z.x_ids],
                u_target[self.mpc_z.u_ids],
            )
        )
        (
            u0[self.mpc_roll.u_ids],
            x_traj[self.mpc_roll.x_ids],
            u_traj[self.mpc_roll.u_ids],
        ) = self.mpc_roll.get_u(
            x0[self.mpc_roll.x_ids],
            x_target[self.mpc_roll.x_ids],
            u_target[self.mpc_roll.u_ids],
        )

        return u0, x_traj, u_traj, t_traj
