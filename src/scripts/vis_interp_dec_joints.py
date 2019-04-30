import pandas as pd
import os
from settings import *
from src.utils.sampu import tsne, encode, load_model, decode
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
sns.set(style="whitegrid")

""" Visualize the decoded joints trajectories for each interpolation. 
    In tsne mode it is done by dimensionality reduction
    In avg mode, average across joints
"""

check_model = '42'
check_epoch = '-200'
mode = 'avg'

# Directory with sampled anims
gen_vae_dir = 'interp_multi_pos'
# All in radians, decoded, normalized
x_dataset = ['11_dec_lerp.csv', '12_dec_slerp.csv', '10_dec_spline.csv'] # Always order as lerp, slerp, spline
# Animation captured from AnimationPlayer in radians
x_naoqi = pd.read_csv('/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/data/naoqi_interp_rec/multi_interp_10-12.csv', index_col=0)

# Plot settings
set_pub()

if mode == 'tsne':

    # Normalization scaler
    scaler_pkl = 'j_scaler_nao_lim_df13_50fps.pkl'
    path = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_pkl)
    scaler = joblib.load(path)
    x_naoqi = scaler.transform(x_naoqi.loc[:, joints_names])
    x_naoqi = pd.DataFrame(columns=joints_names, data=x_naoqi)

    df_tsne = pd.DataFrame()
    fig = plt.figure(figsize=(15, 5))
    ax = plt.axes()

    for data in x_dataset:
        # Load animation dataset
        df = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data), index_col=0)
        name = data.split('_')[0]
        df_tsne[name] = tsne(df, 1)

    df_tsne['naoqi-bezier'] = tsne(x_naoqi, 1)

    df_tsne.plot(ax=ax)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()

else:

    interps_list = []

    for data in x_dataset:
        label = data.split('.')[0].split('_')[2]
        interps_list.append(label)

    df_lerp = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, x_dataset[0]), index_col=0)
    df_slerp = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, x_dataset[1]), index_col=0)
    df_spline = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, x_dataset[2]), index_col=0)

    df_lerp['interp'] = 'VAE Lerp'
    df_slerp['interp'] = 'VAE Slerp'
    df_spline['interp'] = 'VAE B-spline'

    x_naoqi['interp'] = 'NAOqi Bezier'

    df = pd.concat(
        [df_lerp[joints_names + ['interp']], df_slerp[joints_names + ['interp']], df_spline[joints_names + ['interp']],
         x_naoqi[joints_names + ['interp']]], axis=0)

    df = df.pivot(columns='interp')

    # fig, axes = plt.subplots(nrows=17)
    df_mean = df.mean(axis=1, level=1)
    df_mean.columns = df_mean.columns.rename('')
    fig = plt.figure()
    ax = plt.axes()
    df_mean.plot(ax=ax)
    ax.set(xlabel="frames", ylabel="mean joint angles (radians)")

    # To plot each joint separately
    # for i, j in zip(range(17), joints_names):
    #     dfu = df[j]
    #     dfu.plot(title=joints_names[i])
    #
    plt.tight_layout()
    plt.show()




