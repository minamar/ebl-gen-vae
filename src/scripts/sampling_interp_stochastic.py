import pandas as pd
import os
from settings import *
from src.utils.sampu import load_model, decode, sel_anim_id, interpolate
from src.utils.visu import z_anim2dec
from src.data.post_processing import inverse_norm
import seaborn as sns
sns.set(style="darkgrid")
# mina_pal = ["#000000","#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff", "#920000","#924900","#db6d00","#24ff24","#ffff6d"]
# sns.set_palette(mina_pal)
# sns.palplot(sns.color_palette("colorblind", 10))

""" For a specific animation or all animations within the dataset it selects a z_mean posture every interp_steps, 
    adds noise to it (same for different z), and applies interp_steps slerp interpolation between them. 
    The output is decoded to a generated animation(s) and graphs of original +interpolation and decoded are saved 
    in /visualizations.
"""

check_model = '3'
check_epoch = '-500'
x_dataset = 'df14_20fps.csv'
fps = 0.05
z_mean_dataset = '5-500_df14_20fps_mean.csv'
z_sigma_dataset = '5-500_df14_20fps_sigma.csv'
time_lag = 0.02
save_stuff = True
interp_steps = 30
select = True
dataset_dir = x_dataset.split('.')[0]

# Restore model to get the decoder
model = load_model(check_model, check_epoch)

# Load animation dataset and z_means
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
df_z_mean = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_mean_dataset), index_col=0)
df_z_sigma = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_sigma_dataset), index_col=0)

# Latent space dimensions - 2 cols for id and category
z_dim = df_z_mean.shape[1] - 2

eps = 1 + np.random.rand()

# Transform latent
df_z_mean_new = df_z_mean.copy(deep=True)
df_z_mean_new.iloc[:, 0:z_dim] = df_z_mean.iloc[:, 0:z_dim].subtract(df_z_sigma.iloc[:, 0:z_dim].multiply(eps))

if select:
    category = 'Neg/Exc' #'Pos/Exc'
    anim = 'Fearful_1' #'Confident_1'
    # anim, category = sel_anim_id(df_anim)
    anim_id_list = [anim]
else:
    anim_id_list = df_anim['id'].unique().tolist()
    anim_id_list = [x for x in anim_id_list if '_tr' not in x]

for anim_id in anim_id_list:

    df_z_anim = df_z_mean_new.loc[df_z_mean_new['id'] == anim_id, :]
    category = df_z_anim['category'].values[0]
    df_z_anim.drop(columns=['id', 'category'], inplace=True)
    df_z_anim.reset_index(drop=True, inplace=True)

    interp = np.zeros_like(df_z_anim)
    idx_anim = df_z_anim.index.tolist()
    idx_pos = idx_anim[0::interp_steps]
    if idx_anim[-1] - idx_pos[-1] < interp_steps:
        idx_pos[-1] = idx_anim[-1]

    for i in range(len(idx_pos) - 1):
        interp[idx_pos[i]: idx_pos[i + 1], :] = interpolate(df_z_anim.iloc[idx_pos[i], :].values,
                                                            df_z_anim.iloc[idx_pos[i + 1], :].values,
                                                            idx_pos[i + 1] - idx_pos[i])

    interp[idx_pos[-1], :] = df_z_anim.iloc[idx_pos[-1], :].values

    gen_anim = decode(interp, model)
    output_df = pd.DataFrame(columns=joints_names, data=gen_anim)

    # Inverse normalization
    scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
    output_df = inverse_norm(output_df, scaler)
    # Add the time vector
    output_df['time'] = np.linspace(time_lag, (output_df.shape[0] + 1) * fps, output_df.shape[0])
    # Label for the id
    output_df['id'] = 'GEN_' + anim_id
    # Label for the category
    output_df['category'] = category

    fig = z_anim2dec(df_z_mean, interp, output_df, anim_id)

    if save_stuff:
        # Graphs
        plot_path = os.path.join(ROOT_PATH, DATA_VISU, 'interp_nsteps_anim_eps', dataset_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, check_model + check_epoch)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, str(interp_steps) + '_steps')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(os.path.join(plot_path, anim_id + '.eps'), bbox_inches='tight', format='eps',
                    dpi=1000)

        # Dataframes for generated animations
        df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_nsteps_anim_eps', dataset_dir)
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_path = os.path.join(df_path, check_model + check_epoch)
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_path = os.path.join(df_path, str(interp_steps) + '_steps')
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        output_df.to_csv(os.path.join(df_path, anim_id + '.csv'))

