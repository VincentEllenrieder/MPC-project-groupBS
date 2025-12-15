import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    # Q: np.ndarray = np.eye(x_ids.shape[0])
    # R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "y"

    omegax_max  = np.deg2rad(50)
    alpha_max    = np.deg2rad(10)
    vy_max      = 50.0

    d1_max      = np.deg2rad(15)

    Q = np.diag([
        1/(omegax_max**2),
        1/(alpha_max**2),
        2,
    ])

    R = np.array([[1/(d1_max**2) / 10]])
