import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    Q: np.ndarray = np.eye(x_ids.shape[0])
    R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "roll"
