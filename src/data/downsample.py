import numpy as np
import pandas as pd
import os
from settings import *


data_x_set = 'df23_50fps.csv'
dest_x_set = 'df24_20fps.csv'

fps = 3

path = os.path.join(ROOT_PATH, DATA_X_PATH, data_x_set)
df = pd.read_csv(path, index_col=0)

df_new = pd.DataFrame(columns=df.columns)

# A list of all the unique animation id names
names_anim = df.loc[:, 'id'].unique().tolist()

for anim in names_anim:
    # Get a df with all the frames of anim
    df_anim = df.loc[df['id'] == anim, :]
    # TODO: Next line might be the reason for the duplicates on the first frame of each anim
    # df_new.loc[len(df_new), :] = df_anim.iloc[0, :]
    df_new = pd.concat([df_new, df_anim[::fps]], ignore_index=True)

dest = os.path.join(ROOT_PATH, DATA_X_PATH, dest_x_set)
df_new.to_csv(dest)
