import pandas as pd
import os
from settings import *
from src.utils.sampu import load_model, decode, sel_anim_id, interpolate
from src.utils.visu import z_anim2dec
from src.data.post_processing import inverse_norm
import seaborn as sns
sns.set(style="darkgrid")

""" For a specific animation or all animations within the dataset it selects a z_mean posture every interp_steps
    and applies interp_steps slerp interpolation between them. THe output is decoded to a generated animation(s).
"""

check_model = '5'
check_epoch = '-500'
x_dataset = 'df13_50fps.csv'
z_mean_dataset = '5-500_df13_50fps_mean.csv'
z_sigma_dataset = '5-500_df13_50fps_sigma.csv'
time_lag = 0.02
save_stuff = True
interp_steps = 50
select = False

# Restore model to get the decoder
model = load_model(check_model, check_epoch)

# Load animation dataset and z_means
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
df_z_mean = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_mean_dataset), index_col=0)
df_z_sigma = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_sigma_dataset), index_col=0)

# Latent space dimensions - 2 cols for id and category
z_dim = df_z_mean.shape[1] - 2

if select:
    category = 'Pos/Exc'
    anim = 'Confident_1'
    # anim, category = sel_anim_id(df_anim)
    anim_id_list = [anim]
else:
    anim_id_list = df_anim['id'].unique().tolist()
    anim_id_list = [x for x in anim_id_list if '_tr' not in x]

for anim_id in anim_id_list:

    df_z_anim = df_z_mean.loc[df_z_mean['id'] == anim_id, :]
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
    output_df['time'] = np.linspace(time_lag, (output_df.shape[0] + 1) * 0.02, output_df.shape[0])
    # Label for the id
    output_df['id'] = 'GEN_' + anim_id
    # Label for the category
    output_df['category'] = category

    fig = z_anim2dec(df_z_mean, interp, output_df, anim_id, category)

    if save_stuff:
        plot_path = os.path.join(ROOT_PATH, DATA_VISU, 'interp_nsteps_anim', check_model + check_epoch)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_path = os.path.join(plot_path, str(interp_steps) + '_steps')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(os.path.join(plot_path, anim_id + '.eps'), bbox_inches='tight', format='eps',
                    dpi=1000)
        # Save generated
        df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_nsteps_anim', check_model + check_epoch)
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_path = os.path.join(df_path, str(interp_steps) + '_steps')
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        output_df.to_csv(os.path.join(df_path, anim_id + '.csv'))

