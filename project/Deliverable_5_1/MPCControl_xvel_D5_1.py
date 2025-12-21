import numpy as np

from MPCControl_base_D5_1 import MPCControl_base


class MPCControl_xvel_tuned_final(MPCControl_base):
    x_ids = np.array([1, 4, 6])
    u_ids = np.array([1])

    rho_slack = 1e6

    beta_max = np.deg2rad(10)
    d2_max   = np.deg2rad(15)
    wy_max   = np.deg2rad(60)
    vx_max   = 5.0

    Q = np.diag([
        60 * 1/(wy_max**2),
        8 * 1/(beta_max**2),
        60 * 1/(vx_max**2),
    ])

    R = np.array([[(5 * 1/(d2_max**2))]])