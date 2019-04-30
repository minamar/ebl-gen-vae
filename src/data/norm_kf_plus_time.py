import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import os
from settings import *

"""
Takes a keyframe and normalizes the joint values to (0,1).
Additionally, the time dimension is transformed to difference between subsequent steps.
The time dimension is not normalized. 
 
"""

# Original keyframes and destination file to write normalized keyframes
x_set = 'df31_25fps.csv'
dest_x_set = 'df32_25fps.csv'
scaler_file = 'j_scaler_nao_lim_df13_50fps.pkl'  # Change df number with the one of dest_x_set, and whether it is a 'ds' or 'nao'

lims = 'naoqi'  # 'naoqi' to normalize with the full range naoqi limits or 'dataset' to normalize with the dataset range

dest_scaler = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_file)

x_path = os.path.join(ROOT_PATH, DATA_X_PATH, x_set)
df = pd.read_csv(x_path, index_col=0)

# Keep the joint values only
df_joints = df.loc[:, joints_names]

# Normalizes with naoqi full range limits (joints_minmax in settings.py)
if lims == 'naoqi':
    # Add two rows in the end with the min and max values per joint to use it in the normalization
    df_joints.loc[len(df_joints)] = list(joints_minmax[:, 0])
    df_joints.loc[len(df_joints)] = list(joints_minmax[:, 1])

# Train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_joints)

# Normalize the dataset
normalized = scaler.transform(df_joints)

# Save the scaler for inversing the RNN output values later
joblib.dump(scaler, dest_scaler)

# Normalizes with naoqi full range limits (joints_minmax in settings.py)
if lims == 'naoqi':
    # Drop the two last rows containing the min max from the ranges
    normalized = normalized[:-2, :]

# Time dimension transformation: from timestamp to differences between subsequent timestamps
time_s = df.loc[:, 'time']
time_diff = time_s[1:].values - time_s[:-1].values
# Add the missing first element
time_diff = np.insert(time_diff, 0, 0)

# Get the index for the first timestep for each animation
start_anim_idx = []
id_s = df.loc[:, 'id']
anims_l = id_s.unique().tolist()
for l in anims_l:
    start_anim_idx.append(id_s[id_s == l].index[0])

# Use this index to zero out the time_diff's first timestep of each animation
time_diff[np.array(start_anim_idx)] = 0

# Array with the normalized joints and time differences
norm_mat_time = np.c_[normalized, time_diff]
cols = joints_names + ['time_diff']
df_KF_new = pd.DataFrame(data=norm_mat_time, columns=cols)

# Add back anim id column
df_KF_new['id'] = df['id']

# Save to
dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)
df_KF_new.to_csv(dest)
