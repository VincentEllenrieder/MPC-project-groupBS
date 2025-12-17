import numpy as np
import matplotlib.pyplot as plt

from src.rocket import Rocket

from .MPCControl_roll import MPCControl_roll
from .MPCControl_xvel import MPCControl_xvel
from .MPCControl_yvel import MPCControl_yvel
from .MPCControl_zvel import MPCControl_zvel


class MPCVelControl:
    mpc_x: MPCControl_xvel
    mpc_y: MPCControl_yvel
    mpc_z: MPCControl_zvel
    mpc_roll: MPCControl_roll

    def __init__(self) -> None:
        pass

    def new_controller(self, rocket: Rocket, Ts: float, H: float) -> None:
        self.xs, self.us = rocket.trim() # this yields an equilibrium point at the origin (as trim called without x_0 specified yields an equilibrium at origin)
        A, B = rocket.linearize(self.xs, self.us)
        
        self.mpc_x = MPCControl_xvel(A, B, self.xs, self.us, Ts, H)
        self.mpc_y = MPCControl_yvel(A, B, self.xs, self.us, Ts, H)
        self.mpc_z = MPCControl_zvel(A, B, self.xs, self.us, Ts, H)
        self.mpc_roll = MPCControl_roll(A, B, self.xs, self.us, Ts, H)

        return self

    def load_controllers(
        self,
        mpc_x: MPCControl_xvel,
        mpc_y: MPCControl_yvel,
        mpc_z: MPCControl_zvel,
        mpc_roll: MPCControl_roll,
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
        t0: float, # initial timestamp 
        x0: np.ndarray,
        show_Xf: bool = False,
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
        u0[self.mpc_x.u_ids], x_traj[self.mpc_x.x_ids, :], u_traj[self.mpc_x.u_ids, :] = (
            self.mpc_x.get_u(
                x0[self.mpc_x.x_ids],
                show_Xf,
                x_target[self.mpc_x.x_ids],
                u_target[self.mpc_x.u_ids],
            )
        )
        u0[self.mpc_y.u_ids], x_traj[self.mpc_y.x_ids, :], u_traj[self.mpc_y.u_ids, :] = (
            self.mpc_y.get_u(
                x0[self.mpc_y.x_ids],
                show_Xf,
                x_target[self.mpc_y.x_ids],
                u_target[self.mpc_y.u_ids],
            )
        )
        u0[self.mpc_z.u_ids], x_traj[self.mpc_z.x_ids, :], u_traj[self.mpc_z.u_ids, :] = (
            self.mpc_z.get_u(
                x0[self.mpc_z.x_ids],
                show_Xf,
                x_target[self.mpc_z.x_ids],
                u_target[self.mpc_z.u_ids],
            )
        )
        u0[self.mpc_roll.u_ids], x_traj[self.mpc_roll.x_ids, :], u_traj[self.mpc_roll.u_ids, :] = ( 
            self.mpc_roll.get_u(
                x0[self.mpc_roll.x_ids],
                show_Xf,
                x_target[self.mpc_roll.x_ids],
                u_target[self.mpc_roll.u_ids],
            )
        )

        return u0, x_traj, u_traj, t_traj


    def plot_trajectories(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        x_names=None,   # optional list of 12 labels in full-state order
        u_names=None,   # optional list of 4 labels in full-input order
        state_ids=(6, 7, 8, 5),   # v_x, v_y, v_z, roll(gamma)
        input_ids=(0, 1, 2, 3),   # dR, dP, Pavg, Pdiff
        deg_states=(5,),          # which state_ids should be shown in degrees (roll)
        deg_inputs=(0, 1),        # which input_ids should be shown in degrees (dR, dP)
        title_states="Open-loop states",
        title_inputs="Open-loop inputs",
    ):
        """
        x_ol: shape (12, N+1)
        u_ol: shape (4,  N)
        t_ol: shape (N+1,)
        """

        t = np.asarray(t).ravel()
        assert x.ndim == 2 and u.ndim == 2, "x_ol and u_ol must be 2D"
        assert x.shape[1] == t.size, "x_ol and t_ol length mismatch"
        assert u.shape[1] == t.size - 1, "u_ol must have N samples if x_ol has N+1"

        # Default labels if none provided
        if x_names is None:
            x_names = [
                r'$\omega_x$', r'$\omega_y$', r'$\omega_z$',
                r'$\alpha$',   r'$\beta$',   r'$\gamma$',
                r'$v_x$',      r'$v_y$',     r'$v_z$',
                r'$x$',        r'$y$',       r'$z$'
            ]
        if u_names is None:
            u_names = [r'$\delta_1$', r'$\delta_2$', r'$P_{\mathrm{avg}}$', r'$P_{\mathrm{diff}}$']

        # ----------- STATES FIGURE (one axis per state) -----------
        nS = len(state_ids)
        figS, axsS = plt.subplots(1, nS, figsize=(3.6 * nS, 3.2), sharex=True)
        if nS == 1:
            axsS = [axsS]

        for ax, sid in zip(axsS, state_ids):
            y = x[sid, :]
            y_plot = np.rad2deg(y) if sid in deg_states else y
            ax.plot(t, y_plot)
            ax.set_title(x_names[sid])
            ax.grid(True)
            ax.set_xlabel("t [s]")

        axsS[0].set_ylabel("state value")
        figS.suptitle(title_states)
        figS.tight_layout()

        # ----------- INPUTS FIGURE (one axis per input) -----------
        nU = len(input_ids)
        figU, axsU = plt.subplots(1, nU, figsize=(3.6 * nU, 3.2), sharex=True)
        if nU == 1:
            axsU = [axsU]

        for ax, uid in zip(axsU, input_ids):
            y = u[uid, :]
            y_plot = np.rad2deg(y) if uid in deg_inputs else y
            ax.step(t[:-1], y_plot, where="post")
            ax.set_title(u_names[uid])
            ax.grid(True)
            ax.set_xlabel("t [s]")

        axsU[0].set_ylabel("input value")
        figU.suptitle(title_inputs)
        figU.tight_layout()

        plt.show()

