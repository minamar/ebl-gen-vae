import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import os
from settings import *

"""
For dataframes with motion + LEDs. Normalizes the joint values to (0,1).
Ignores the time and the LEDs columns. 
"""

# Original keyframes and destination file to write normalized keyframes
x_set = 'df31_25fps.csv'
dest_x_set = 'df32_25fps.csv'
# Change df number with the one of dest_x_set, and whether it is a 'ds' or 'nao'
scaler_file = 'j_scaler_nao_lim_df32_25fps.pkl'
# 'naoqi' to normalize with the full range naoqi limits or 'dataset' to normalize with the dataset range
lims = 'naoqi'

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

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_joints)
normalized = scaler.transform(df_joints)
joblib.dump(scaler, dest_scaler)

# Normalizes with naoqi full range limits (joints_minmax in settings.py)
if lims == 'naoqi':
    # Drop the two last rows containing the min max from the ranges
    normalized = normalized[:-2, :]

# Array with the normalized joints added to the initial dataframe
df[joints_names] = normalized

# Save to
dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)
df.to_csv(dest)
