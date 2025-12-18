import numpy as np

from src.rocket import Rocket

from MPCControl_xvel_D4_1 import MPCControl_xvel_tuned_final
from MPCControl_yvel_D4_1 import MPCControl_yvel_tuned_final
from MPCControl_zvel_D4_1 import MPCControl_zvel_tuned_final
from MPCControl_roll_D4_1 import MPCControl_roll_tuned_final


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

        if x_target is None:
            x_target = self.xs

        if u_target is None:
            u_target = self.us
        
        # Extract scalar references from full state target
        vx_ref    = x_target[6]
        vy_ref    = x_target[7]
        vz_ref    = x_target[8]
        gamma_ref = x_target[5]

        # --- Saturate velocity references to feasible region ---
        vx_ref = np.clip(vx_ref, -5.0, 5.0)
        vy_ref = np.clip(vy_ref, -5.0, 5.0)
        vz_ref = np.clip(vz_ref, -5.0, 5.0)

        gamma_ref = np.clip(gamma_ref,
                            -np.deg2rad(45),
                            np.deg2rad(45))

        # ---- X velocity ----
        # (
        #     u0[self.mpc_x.u_ids],
        #     x_traj[self.mpc_x.x_ids],
        #     u_traj[self.mpc_x.u_ids],
        # ) = self.mpc_x.get_u(
        #     x0[self.mpc_x.x_ids],
        #     x_target=vx_ref,
        # )
        try:
            (
                u0[self.mpc_x.u_ids],
                x_traj[self.mpc_x.x_ids],
                u_traj[self.mpc_x.u_ids],
            ) = self.mpc_x.get_u(
                x0[self.mpc_x.x_ids],
                x_target=vx_ref,
            )
        except RuntimeError as e:
            print(f"\n=== FAILED: mpc_x at t={t0:.2f} ===")
            print("x0_sub:", x0[self.mpc_x.x_ids])
            print("vx_ref:", vx_ref)
            raise


        # ---- Y velocity ----
        # (
        #     u0[self.mpc_y.u_ids],
        #     x_traj[self.mpc_y.x_ids],
        #     u_traj[self.mpc_y.u_ids],
        # ) = self.mpc_y.get_u(
        #     x0[self.mpc_y.x_ids],
        #     x_target=vy_ref,
        # )

        try:
            (
                u0[self.mpc_y.u_ids],
                x_traj[self.mpc_y.x_ids],
                u_traj[self.mpc_y.u_ids],
            ) = self.mpc_y.get_u(
                x0[self.mpc_y.x_ids],
                x_target=vy_ref,
            )
        except RuntimeError as e:
            print(f"\n=== FAILED: mpc_y at t={t0:.2f} ===")
            print("x0_sub:", x0[self.mpc_y.x_ids])
            print("vy_ref:", vy_ref)
            raise

        # ---- Z velocity ----
        # (
        #     u0[self.mpc_z.u_ids],
        #     x_traj[self.mpc_z.x_ids],
        #     u_traj[self.mpc_z.u_ids],
        # ) = self.mpc_z.get_u(
        #     x0[self.mpc_z.x_ids],
        #     x_target=vz_ref,
        # )

        try:
            (
                u0[self.mpc_z.u_ids],
                x_traj[self.mpc_z.x_ids],
                u_traj[self.mpc_z.u_ids],
            ) = self.mpc_z.get_u(
                x0[self.mpc_z.x_ids],
                x_target=vz_ref,
            )
        except RuntimeError as e:
            print(f"\n=== FAILED: mpc_z at t={t0:.2f} ===")
            print("x0_sub:", x0[self.mpc_z.x_ids])
            print("vz_ref:", vz_ref)
            raise

        # ---- Roll ----
        # (
        #     u0[self.mpc_roll.u_ids],
        #     x_traj[self.mpc_roll.x_ids],
        #     u_traj[self.mpc_roll.u_ids],
        # ) = self.mpc_roll.get_u(
        #     x0[self.mpc_roll.x_ids],
        #     x_target=gamma_ref,
        # )

        # ---- Roll ----
        try:
            (
                u0[self.mpc_roll.u_ids],
                x_traj[self.mpc_roll.x_ids],
                u_traj[self.mpc_roll.u_ids],
            ) = self.mpc_roll.get_u(
                x0[self.mpc_roll.x_ids],
                x_target=gamma_ref,
            )
        except RuntimeError as e:
            print(f"\n=== FAILED: mpc_roll at t={t0:.2f} ===")
            print("x0_sub:", x0[self.mpc_roll.x_ids])
            print("gamma_ref:", gamma_ref)
            raise

        return u0, x_traj, u_traj, t_traj
