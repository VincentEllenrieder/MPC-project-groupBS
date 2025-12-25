import numpy as np

from MPCControl_base_D6_2 import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    alpha_max = np.deg2rad(10)
    d1_max   = np.deg2rad(15)
    wx_max = np.deg2rad(60)
    vy_max = 5.0

    Q = np.diag([
        60 * 1/(wx_max**2),
        8 *1/(alpha_max**2),
        60 * 1/(vy_max**2),
        50
    ])

    R = np.array([[(5 * 1/(d1_max**2))]])

    subsys_name = 'y'
