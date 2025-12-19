import numpy as np

from MPCControl_base_D5_1 import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg
    tracked_idx = 0   # only state is v_z

class MPCControl_zvel_tuned_default(MPCControl_zvel):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 5.0
    dP_max = min(80 - 66.7, 66.7 - 40.0)  # m/s per control step

    Q = np.array([[1/(vz_max**2)]])
    R = np.array([[1/(dP_max**2)]])

class MPCControl_zvel_tuned_final(MPCControl_zvel):
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    vz_max = 5.0
    dP_max = min(80 - 66.7, 66.7 - 40.0)  # m/s per control step


    Q = np.array([[20 * (1/(vz_max**2))]])
    R = np.array([[(1/(dP_max**2)) / 2]])


