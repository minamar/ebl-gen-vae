import os
from sklearn.externals import joblib
from settings import *

""" 
Normalize the standInit posture. 
Script must be run after norm_kf_plus_time, so that it can use the scaler generated from it 
"""

scaler_pkl = 'j_scaler_nao_lim_df13_50fps.pkl'

path = os.path.join(ROOT_PATH, 'src/data', scaler_pkl)
scaler = joblib.load(path)


standInit_joints = np.zeros([1, n_joints])
standInit_joints[0, :] = standInit


normalized = scaler.transform(standInit_joints)
normalized = normalized.tolist()
normalized = [ np.round(x, 7) for x in normalized[0]]
standInit_norm = normalized
print(standInit_norm)