import numpy as np

from MPCControl_base_D5_1 import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7]) #wx, alpha, vy
    u_ids: np.ndarray = np.array([0])  # d1
    tracked_idx = 2   # v_y

class MPCControl_yvel_tuned_default(MPCControl_yvel):
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

class MPCControl_yvel_tuned_final(MPCControl_yvel):
    x_ids: np.ndarray = np.array([0, 3, 7]) #wx, alpha, vy
    u_ids: np.ndarray = np.array([0])  # d1

    rho_slack = 1e6

    alpha_max = np.deg2rad(10)
    d1_max   = np.deg2rad(15)
    wx_max = np.deg2rad(60)
    vy_max = 5.0

    Q = np.diag([
        60 * 1/(wx_max**2),
        8 *1/(alpha_max**2),
        60 * 1/(vy_max**2),
    ])

    R = np.array([[(5 * 1/(d1_max**2))]])
    

