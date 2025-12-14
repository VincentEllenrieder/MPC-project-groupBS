import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    Q: np.ndarray = np.eye(x_ids.shape[0])
    R: np.ndarray = np.eye(u_ids.shape[0])

    subsys_name: str = "y"
