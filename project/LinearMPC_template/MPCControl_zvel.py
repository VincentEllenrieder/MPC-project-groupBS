import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    Q: np.ndarray = np.array([[1]])
    R: np.ndarray = np.array([[1]])

    subsys_name: str = "z"