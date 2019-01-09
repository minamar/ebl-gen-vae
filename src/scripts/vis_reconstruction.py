import pandas as pd
import os
from settings import *
from src.utils.visu import x2reco
from src.data.post_processing import inverse_norm

""" For a given model and dataset save the plots of the original joints and their reconstruction (.- linestyle) 
    over time for each animation. Top one is for head and torso, middle one for right arm, botom for left arm.  
"""

check_model = '1'
check_epoch = '-500'
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
x_dataset = 'df14_20fps.csv'
z_dataset = check_model + check_epoch + '_df14_20fps_mean.csv'
reco_dataset = check_model + check_epoch + '_all.csv'
dataset_dir = x_dataset.split('.')[0]
save_stuff = True

# Load animation original dataset
df_x = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
# Load latent z_mean for d_x
df_z_mean = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_dataset), index_col=0)
# Load reconstructed
df_reco = pd.read_csv(os.path.join(ROOT_PATH, DATA_RECO, dataset_dir, check_model + check_epoch, reco_dataset), index_col=0)

anim_id_list = df_x['id'].unique().tolist()
anim_id_list = [x for x in anim_id_list if '_tr' not in x]

for anim_id in anim_id_list:
    df_x_anim = df_x.loc[df_x['id'] == anim_id, :]
    df_x_anim.reset_index(inplace=True, drop=True)
    df_x_anim = inverse_norm(df_x_anim.loc[:, joints_names], scaler)

    df_z_mean_anim = df_z_mean.loc[df_z_mean['id'] == anim_id, :]
    df_z_mean_anim.reset_index(inplace=True, drop=True)

    df_reco_anim = df_reco.loc[df_reco['id'] == anim_id, :]
    df_reco_anim.reset_index(inplace=True, drop=True)

    fig = x2reco(df_x_anim, df_reco_anim)

    if save_stuff:
        plot_path = os.path.join(ROOT_PATH, DATA_VISU, 'reconstruction', dataset_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, check_model + check_epoch)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(os.path.join(plot_path, anim_id + '.eps'), bbox_inches='tight', format='eps', dpi=1000)

