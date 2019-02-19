import pandas as pd
import os
from settings import *
from src.utils.sampu import tsne, encode, load_model, decode
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from mpl_toolkits import mplot3d
import seaborn as sns
sns.set(style="darkgrid")

check_model = '42'
check_epoch = '-200'
# mode = 'tsne' # Dim reduction on the decoded latent interpolants (normalized) and choregraphe trajectory
mode = 'latent' # No dim reduction. Vis latent interpolants vs encoded choregraph trajectory

# All in radians, decoded, normalized
x_dataset = [ 'slerp_42-200_Loving_01_465.csv', 'lerp_42-200_Loving_01_465.csv', 'bezier_42-200_Loving_01_465.csv']

# All in radians, decoded, normalized
z_dataset = ['slerp_z_42-200_Loving_01_465.csv', 'lerp_z_42-200_Loving_01_465.csv', 'bezier_z_42-200_Loving_01_465.csv']

# Animation captured from AnimationPlayer in radians
x_naoqi = pd.read_csv('/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/data/naoqi_interp_rec/465_Loving_01.csv', index_col=0)

# Normalization scaler
scaler_pkl = 'j_scaler_nao_lim_df13_50fps.pkl'
path = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_pkl)
scaler = joblib.load(path)
x_naoqi = scaler.transform(x_naoqi.loc[:, joints_names])
x_naoqi = pd.DataFrame(columns=joints_names, data=x_naoqi)
# Plot settings
set_pub()

if mode == 'tsne':
    df_tsne = pd.DataFrame()
    fig = plt.figure(figsize=(15, 5))
    ax = plt.axes()

    for data in x_dataset:
        # Load animation dataset
        df = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, 'interp_2postures', data), index_col=0)
        name = data.split('_')[0]
        df_tsne[name] = tsne(df, 1)

    df_tsne['radian-bezier'] = tsne(x_naoqi, 1)

    df_tsne.plot(ax=ax)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()

elif mode == 'latent':
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Encode the normalized naoqi
    # Restore model to get the decoder
    model = load_model(check_model, check_epoch)
    latent_mean, latent_sigma = encode(x_naoqi.loc[:, joints_names], model)

    df_z_all = pd.DataFrame(latent_mean)
    df_z_all.columns = ['l1', 'l2', 'l3']
    df_z_all['interp'] = 'naoqi-bezier'
    ax.plot(df_z_all['l1'], df_z_all['l2'], df_z_all['l3'], label='naoqi-bezier')

    # # ======= Subsequent enc-dec-enc
    # df_dec_slerp = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, 'interp_2postures', 'slerp_42-200_Loving_01_465.csv'), index_col=0)
    # latent_mean, latent_sigma = encode(df_dec_slerp.loc[:, joints_names], model)
    #
    # df_dec_slerp = pd.DataFrame(latent_mean)
    # df_dec_slerp.columns = ['l1', 'l2', 'l3']
    # df_dec_slerp['interp'] = 'naoqi-bezier'
    # ax.plot(df_dec_slerp['l1'], df_dec_slerp['l2'], df_dec_slerp['l3'], label='dec_slerp')
    #
    # df_dec_slerp2 = decode(latent_mean, model)
    # df_dec_slerp2 = pd.DataFrame(columns=joints_names, data=df_dec_slerp2)
    # latent_mean, latent_sigma = encode(df_dec_slerp2.loc[:, joints_names], model)
    # df_dec_slerp2 = pd.DataFrame(latent_mean)
    # df_dec_slerp2.columns = ['l1', 'l2', 'l3']
    # df_dec_slerp2['interp'] = 'naoqi-bezier'
    # ax.plot(df_dec_slerp2['l1'], df_dec_slerp2['l2'], df_dec_slerp2['l3'], label='dec2_slerp')
    # # ========

    for data in z_dataset:
        # Load animation dataset
        df = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, 'interp_2postures', data), index_col=0)
        df.columns = ['l1', 'l2', 'l3', 'interp']
        df['interp'] = 'vae_' + data.split('_')[0]
        ax.plot(df['l1'], df['l2'], df['l3'], label= 'vae_' + data.split('_')[0])
        # df_z_all = pd.concat([df_z_all, df], axis=0).reset_index(drop=True)

    # ax.plot(df_z_all['l1'], df_z_all['l2'], df_z_all['l3'], label=df_z_all['interp'])
    # df_z_all.plot(ax=ax, value_name='interp')
    ax.set_xlabel('l1')
    ax.set_ylabel('l2')
    ax.set_zlabel('l3')

    ax.set_title('Interpolants in the latent space')
    # ax.axis('equal')
    # ax.axis('square')
    ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(0.96, 0.5))

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()

else:
    pass



