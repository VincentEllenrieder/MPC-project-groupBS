import numpy as np

from MPCControl_base_D5_1 import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])
    tracked_idx = 1   # gamma

class MPCControl_roll_tuned_default(MPCControl_roll):
    x_ids = np.array([2, 5])
    u_ids = np.array([3])

    wz_max    = np.deg2rad(120)   # acceptable roll rate
    gamma_max = np.deg2rad(30)    # per spec
    pdiff_max = 20.0              # [%]

    Q = np.diag([
        1/(wz_max**2),
        1/(gamma_max**2),
    ])

    R = np.array([[1/(pdiff_max**2)]])

class MPCControl_roll_tuned_final(MPCControl_roll):
    x_ids = np.array([2, 5])
    u_ids = np.array([3])

    wz_max    = np.deg2rad(120)
    gamma_max = np.deg2rad(30)
    pdiff_max = 20.0

    Q = np.diag([
        1/(wz_max**2),
        0.75 * 1/(gamma_max**2),
    ])

    R = np.array([[1/(pdiff_max**2) * 1.5]])

