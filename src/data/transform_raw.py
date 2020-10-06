import numpy as np
import pandas as pd
import os
from settings import *

"""
Takes as input the dataframe with the raw data collected from the .xar files and transforms it:
- Keyframes to timestamps (keyframe number * 1/fps)
- Joint values to radians from degrees (all joints except LHand, RHand)
- Keyframes shifted forward by 20
- Init posture added at 1 = 0.04 sec
"""

# Original keyframes and destination file to write normalized keyframes
data_raw = 'df10_KF.csv'
dest_x_set = 'df11_KF.csv'

# Path to get raw and for the destination
path_raw = os.path.join(ROOT_PATH, RAW_DATA, data_raw)
dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)

# Load raw data
df_raw = pd.read_csv(path_raw, index_col=0)

# Print some stuff for confirmation
# First keyframe, last keyframe, #keyframes (all stored in data/external/plymouth_animations_descrptive.ods)
print(df_raw.groupby('id', )['keyframe'].agg(['first', 'last', 'count']))

# Convert degrees to radians for every joint EXCEPT LHand, RHand (since their values are not in degrees)
j_cols_deg = [0,1,2,3,4,5,6,8,9,10,11,12,14,15,16]
df_raw.iloc[:, j_cols_deg] = df_raw.iloc[:, j_cols_deg].apply(lambda x: np.radians(x))

# # Shift keyframes forward by 20
df_raw['keyframe'] = df_raw['keyframe'].apply(lambda x: x + 20)

# Convert keyframes to timestamp in secs for all animations except 'Happy_4'
fps1 = 25
df_raw.loc[df_raw['id'] != 'Happy_4', 'keyframe'] = df_raw.loc[df_raw['id'] != 'Happy_4', 'keyframe'].apply(lambda x: np.round(x * (1 / fps1), 2))

# Convert keyframes to timestamp in secs only 'Happy_4'
fps2 = 15
df_raw.loc[df_raw['id'] == 'Happy_4', 'keyframe'] = df_raw.loc[df_raw['id'] == 'Happy_4', 'keyframe'].apply(lambda x: np.round(x * (1 / fps2), 2))

# Add Init posture at the first keyframe (0.04 secs)

# Get animations ids
id = list(df_raw['id'].unique())

# Create init dataframe
df_init = pd.DataFrame(columns=joints_names+['keyframe', 'id'])
df_init['id'] = id
df_init['keyframe'] = 0.04
df_init.loc[:, :-2] = standInit
#
df_trans = df_raw.append(df_init)
# df_trans = df_raw
df_trans.sort_values(by=['id','keyframe'], inplace=True)
df_trans.reset_index(drop=True, inplace=True)

# Change column name 'keyframe' to 'time'
df_trans.rename(columns={'keyframe': 'time'}, inplace=True)

# Save transformed dataframe
df_trans.to_csv(dest)


