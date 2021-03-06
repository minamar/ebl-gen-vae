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
sns.set(style="darkgrid")

""" Visualize the decoded joints trajectories for different radius on a longitudinal interpolation grid style. 
    In avg mode, average across joints
"""

check_model = '42'
check_epoch = '-200'
mode = 'avg'

# Directory with sampled anims
gen_vae_dir = 'interp_grid'

re1 = r'(1[0-1]|[0-9])_(dec)'
re2 = r'(1[2-9]|2[0-3])_(dec)'
re3 = r'(2[4-9]|3[0-5])_(dec)'
re_list = [re1, re2, re3]

path_to_folder = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir)
with open(path_to_folder + '/-overview.json', 'r') as f:
    details = json.load(f)
# Plot settings
set_pub()
fig, ax = plt.subplots(3, sharex=True, figsize=(30, 18))

for i in range(3):
    # itemList = os.listdir(path_to_folder)
    x_dataset = [f for f in os.listdir(path_to_folder) if re.match(re_list[i], f)]
    # All in radians, decoded, normalized

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
    df_plot = df_mean # .loc[0:500]
    df_plot.plot(sort_columns=True, colormap='gist_heat', ax=ax[i], yticks=np.arange(-0.4, 0.6, 0.2), xticks=np.arange(0, df_plot.shape[0], 100), legend=False)

labels = [item.get_text() for item in ax[2].get_xticklabels()]
empty_string_labels = [str(x) for x in list(np.arange(10))]  #['']*len(labels)
ax[2].set_xticklabels(empty_string_labels)
ax[2].set(xlabel="longitude lines")
ax[1].set(ylabel="mean joint angles (radians)")
ax[1].legend(title='radius', loc='center left', bbox_to_anchor=(0.96, 0.5))

plt.tight_layout()

plt.show()
