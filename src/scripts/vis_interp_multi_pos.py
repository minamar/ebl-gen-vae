import pandas as pd
import os
from settings import *
from src.utils.sampu import tsne, encode, load_model, decode
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

""" Visualize the latent interpolants sampled by interp_multi_pos"""

check_model = '42-200'
check_epoch = '-200'


# Directory with sampled anims
gen_vae_dir = 'interp_multi_pos'
# All in radians, decoded, normalized
x_dataset = ['15_dec_spline.csv', '17_dec_lerp.csv', '16_dec_slerp.csv']
# All in latent space
z_dataset = ['15_z_spline.csv', '17_z_lerp.csv', '16_z_slerp.csv']
# Animation captured from AnimationPlayer in radians
x_naoqi = pd.read_csv('.../EXP3_Generation/data/naoqi_interp_rec/465_Loving_01.csv', index_col=0)

# Normalization scaler
scaler_pkl = 'j_scaler_nao_lim_df13_50fps.pkl'
path = os.path.join(ROOT_PATH, SCALERS_PATH, scaler_pkl)
scaler = joblib.load(path)
x_naoqi = scaler.transform(x_naoqi.loc[:, joints_names])
x_naoqi = pd.DataFrame(columns=joints_names, data=x_naoqi)

# Plot settings
set_pub()


def update_plot(frame_number, zarray, plot):
    plot.set_data(zarray[:2, :frame_number])
    plot.set_3d_properties(zarray[2, :frame_number])


fig = plt.figure()
ax = plt.axes(projection='3d')

    # Repeatedly encode the decoded to see how the generated animation is changing
    # # Encode the normalized naoqi
    # model = load_model(check_model, check_epoch)
    # latent_mean, latent_sigma = encode(x_naoqi.loc[:, joints_names], model)
    #
    # df_z_all = pd.DataFrame(latent_mean)
    # df_z_all.columns = ['l1', 'l2', 'l3']
    # df_z_all['interp'] = 'naoqi-bezier'
    # ax.plot(df_z_all['l1'], df_z_all['l2'], df_z_all['l3'], label='naoqi-bezier')

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
    # Load the z interpolants
    df = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data), index_col=0)

    label = data.split('.')[0].split('_')[2]
    ax.plot(df['l1'], df['l2'], df['l3'], label=label)

    # Animated plot
    # zarray = df.loc[:, ['l1', 'l2', 'l3']].values.transpose()
    # plot, = ax.plot(zarray[0, 0:1], zarray[1, 0:1], zarray[2, 0:1])
    # # Setting the axes properties
    # ax.set_xlim3d([zarray[0].min(), zarray[0].max()])
    # ax.set_ylim3d([zarray[0].min(), zarray[0].max()])
    # ax.set_zlim3d([zarray[0].min(), zarray[0].max()])

ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')

# ax.set_title('Interpolants in the latent space')
# ax.axis('equal')
# ax.axis('square')
ax.auto_scale_xyz
ax.legend()

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

# ani = animation.FuncAnimation(fig, update_plot, zarray.shape[1], fargs=(zarray, plot), interval=100, blit=False)

plt.show()





