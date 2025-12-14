import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    Q: np.ndarray = np.eye(x_ids.shape[0])
    R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "x"
