import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    # Q: np.ndarray = np.eye(x_ids.shape[0])
    # R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "roll"

    omegaz_max  = np.deg2rad(50)
    gamma_max   = np.deg2rad(50)

    Pdiff_max   = 20.0

    Q = np.diag([
        1/(omegaz_max**2),
        2 * 1/(gamma_max**2),
    ])

    R = np.array([[1/(Pdiff_max**2) * 1.5 ]])
