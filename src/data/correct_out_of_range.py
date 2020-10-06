import pandas as pd
import os
from settings import *

"""
Detects joint values that exceed the limits defined in joints_minmax and replaces them with the min max limits
"""

# Original keyframes and destination file
x_set = 'df1_50fps.csv'
dest_x_set = 'df11_50fps.csv'


path = os.path.join(ROOT_PATH, RAW_DATA, x_set)
df = pd.read_csv(path, index_col=0)


j_count = 0
for joint in joints_names:
    min = joints_minmax[j_count][0]
    max = joints_minmax[j_count][1]
    idx_min = df[joint].loc[df[joint] < min].index
    df.loc[idx_min,joint] = joints_minmax[j_count][0]
    idx_max = df[joint].loc[df[joint] > max].index
    df.loc[idx_max,joint] = joints_minmax[j_count][1]
    j_count += 1

# Save to
dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)
df.to_csv(dest)
