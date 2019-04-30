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
lat_dim = ['l1', 'l2', 'l3']
# Directory with sampled anims
gen_vae_dir = 'interp_grid_longitude/42-200'


fig, ax = plt.subplots(3, sharex=True, figsize=(30, 18))
# 3 plots, one for each latent dim
set_pub()
for l in range(3):
    path_to_folder = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, lat_dim[l])
    with open(path_to_folder + '/-overview.json', 'r') as f:
        details = json.load(f)

# re1 = r'(1[0-1]|[0-9])_(dec)'
# re2 = r'(1[2-9]|2[0-3])_(dec)'
# re3 = r'(2[4-9]|3[0-5])_(dec)'
# re_list = [re1, re2, re3]

    re1 = r'\w+(dec)+\w'

    filenames = [f for f in os.listdir(path_to_folder) if re.match(re1, f)]
    # All in radians, decoded, normalized
    x_dataset = sorted(filenames, key=lambda item: (int(item.partition('_')[0])
                                                    if item[0].isdigit() else float('inf'), item))
    df = pd.DataFrame()
    for data in x_dataset:
        label = data.split('_')[0]
        radius = details[label]['grid_radius']
        longitude = details[label]['longitude']
        df_data = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, lat_dim[l], data), index_col=0)
        df_data.drop(columns='time', inplace=True)
        df_data['radius'] = radius
        # df_data['longitude'] = longitude

        df = pd.concat([df, df_data], axis=0)

    df.reset_index(drop=True, inplace=True)  # without it, pivot finds duplicates
    idx = list(np.arange(0, 2000, 1)) * 10      # bad solution, but pivot without subindex per radius gives NaNs
    df = df.pivot(columns='radius', index=idx)

    df_mean = df.mean(axis=1, level=1)
    df_mean.columns = df_mean.columns.rename('')
    df_plot = df_mean

    df_plot.plot(sort_columns=True, colormap='gist_heat', ax=ax[l], yticks=np.arange(-0.1, 0.4, 0.2), xticks=np.arange(0, df_plot.shape[0], 200), legend=False)

ax[0].text(1, 0.35, 'LD1',  weight='bold', fontsize=14)
ax[1].text(1, 0.35, 'LD2',  weight='bold', fontsize=14)
ax[2].text(1, 0.35, 'LD3',  weight='bold', fontsize=14)

labels = [item.get_text() for item in ax[2].get_xticklabels()]
empty_string_labels = [str(x) for x in list(np.arange(10))]
ax[2].set_xticklabels(empty_string_labels)
ax[2].set(xlabel="longitude line")
ax[1].set(ylabel="mean joint angle (radians)")
ax[1].legend(title='radius', loc='center left', bbox_to_anchor=(0.96, 0.5))

plt.tight_layout()

plt.show()

# Frequency analysis
# df_l1 = df_mean
# df_l2 = df_mean
# df_l3 = df_mean
#
# fig4 = plt.figure()
# sp = np.fft.fft(df_l1.iloc[:, 4])
# samples = sp.shape[-1]-1
# freq = np.fft.fftfreq(samples, 0.05)
# plt.plot(freq[:int(samples/2)], (np.abs(sp.real))[:int(samples/2)], label='l1')
#
# sp = np.fft.fft(df_l2.iloc[:, 4])
# samples = sp.shape[-1]-1
# freq = np.fft.fftfreq(samples, 0.05)
# plt.plot(freq[:int(samples/2)], (np.abs(sp.real))[:int(samples/2)], label='l2')
#
# sp = np.fft.fft(df_l3.iloc[:, 4])
# samples = sp.shape[-1]-1
# freq = np.fft.fftfreq(samples, 0.05)
# plt.plot(freq[:int(samples/2)], (np.abs(sp.real))[:int(samples/2)], label='l3')
#
# plt.xlim([0, 2])
# plt.xticks(np.arange(0, 2, step=0.2))
# plt.ylim([0, 50])
