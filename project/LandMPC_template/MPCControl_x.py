import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    beta_max = np.deg2rad(10)
    d2_max   = np.deg2rad(15)
    wy_max   = np.deg2rad(60)
    vx_max   = 5.0

    Q = np.diag([
        60 * 1/(wy_max**2),
        8 * 1/(beta_max**2),
        60 * 1/(vx_max**2),
        50
    ])

    R = np.array([[(5 * 1/(d2_max**2))]])

    subsys_name = 'x'
