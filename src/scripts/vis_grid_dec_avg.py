import pandas as pd
import os
import json
import glob
import re
from settings import *
from src.utils.sampu import tsne, encode, load_model, decode
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
sns.set(style="whitegrid")

""" Visualize the decoded joints trajectories for different radius on a longitudinal interpolation grid style. 
    In tsne mode it is done by dimensionality reduction
    In avg mode, average across joints
"""

check_model = '42'
check_epoch = '-200'
mode = 'avg'

# Directory with sampled anims
gen_vae_dir = 'interp_grid'

re1 = r'(1[0-1]|[0-9])_(dec)'
re2 = r'(1[2-9]|2[0-3])_(dec)'

path_to_folder = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir)
# itemList = os.listdir(path_to_folder)
itemList = [f for f in os.listdir(path_to_folder) if re.match(re2, f)]
# All in radians, decoded, normalized
x_dataset = [item for item in itemList if "dec" in item]

with open(path_to_folder + '/-overview.json', 'r') as f:
    details = json.load(f)

# Plot settings
set_pub()

df = pd.DataFrame()
for data in x_dataset:
    label = data.split('_')[0]
    radius = details[label]['grid_radius']
    df_data = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data), index_col=0)
    df_data.drop(columns='time', inplace=True)
    # df_data['set'] = label
    df_data['radius'] = radius

    df = pd.concat([df, df_data], axis=0)

df = df.pivot(columns='radius')

df_mean = df.mean(axis=1, level=1)
df_mean.columns = df_mean.columns.rename('')
ax = df_mean.plot(title='Decoded joints trajectories', sort_columns=True, colormap='gist_heat')
ax.set(xlabel="frames", ylabel="mean joint angles (radians)")
ax.legend(title='radius')

# Plot each joint separately
# for i, j in zip(range(17), joints_names):
#     dfu = df[j]
#     dfu.plot(title=joints_names[i])
#
plt.tight_layout()
plt.show()
