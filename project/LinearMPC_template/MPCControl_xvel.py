import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    # Q: np.ndarray = np.eye(x_ids.shape[0])
    # R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "x"

    omegay_max  = np.deg2rad(50)
    beta_max    = np.deg2rad(10)
    vx_max      = 50.0

    d2_max      = np.deg2rad(15)

    Q = np.diag([
        1/(omegay_max**2),
        1/(beta_max**2),
        2,
    ])

    R = np.array([[1/(d2_max**2) / 10]])
