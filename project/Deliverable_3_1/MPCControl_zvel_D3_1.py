import numpy as np

from MPCControl_base_D3_1 import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

class MPCControl_zvel_tuned_default(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 10.0
    P_max = 80

    Q = np.array([[1/(vz_max**2)]])
    R = np.array([[1/(P_max**2)]])

class MPCControl_zvel_tuned_final(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 10.0
    P_max = 80

    Q = np.array([[80 * (1/(vz_max**2))]])
    R = np.array([[50 * (1/(P_max**2))]])


