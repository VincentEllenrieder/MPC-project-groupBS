import numpy as np

from MPCControl_base_D3_3 import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg
    tracked_idx = 0   # only state is v_z

class MPCControl_zvel_tuned_default(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 5.0
    P_max = 80

    Q = np.array([[1/(vz_max**2)]])
    R = np.array([[1/(P_max**2)]])

class MPCControl_zvel_tuned_final(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 10.0
    P_max = 80

    Q = np.array([[100 * (1/(vz_max**2))]])
    R = np.array([[50 * (1/(P_max**2))]])


