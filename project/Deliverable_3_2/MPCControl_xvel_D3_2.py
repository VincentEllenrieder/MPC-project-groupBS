import numpy as np

from MPCControl_base_D3_2 import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])     #wx ,beta,vx
    u_ids: np.ndarray = np.array([1])   # d2
    tracked_idx = 2   # v_x

class MPCControl_xvel_tuned_default(MPCControl_xvel):
    x_ids = np.array([1, 4, 6])
    u_ids = np.array([1])

    beta_max = np.deg2rad(10)
    d2_max   = np.deg2rad(15)
    wy_max   = np.deg2rad(60)
    vx_max   = 5.0

    Q = np.diag([
        1/(wy_max**2),
        1/(beta_max**2),
        1/(vx_max**2),
    ])

    R = np.array([[1/(d2_max**2)]])

class MPCControl_xvel_tuned_final(MPCControl_xvel):
    x_ids = np.array([1, 4, 6])
    u_ids = np.array([1])

    beta_max = np.deg2rad(10)
    d2_max   = np.deg2rad(15)
    wy_max   = np.deg2rad(60)
    vx_max   = 5.0

    Q = np.diag([
        1/(wy_max**2),
        1/(beta_max**2),
        40 * 1/(vx_max**2),
    ])

    R = np.array([[(1/(d2_max**2)) / 10]])
