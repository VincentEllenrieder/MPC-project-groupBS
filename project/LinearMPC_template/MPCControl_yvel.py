import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7]) #wx, alpha, vy
    u_ids: np.ndarray = np.array([0])  # d1

class MPCControl_yvel_tuned_default(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7]) #wx, alpha, vy
    u_ids: np.ndarray = np.array([0])  # d1

    alpha_max = np.deg2rad(10)
    d1_max   = np.deg2rad(15)
    wx_max = np.deg2rad(60)
    vy_max = 5.0

    Q = np.diag([
        1/(wx_max**2),
        1/(alpha_max**2),
        1/(vy_max**2),
    ])

    R = np.array([[1/(d1_max**2)]])

class MPCControl_yvel_tuned_final(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7]) #wx, alpha, vy
    u_ids: np.ndarray = np.array([0])  # d1

    alpha_max = np.deg2rad(10)
    d1_max   = np.deg2rad(15)
    wx_max = np.deg2rad(60)
    vy_max = 5.0

    Q = np.diag([
        1/(wx_max**2),
        1/(alpha_max**2),
        40 * 1/(vy_max**2),
    ])

    R = np.array([[(1/(d1_max**2)) / 10]])


