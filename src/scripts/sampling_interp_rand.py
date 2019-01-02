import pandas as pd
import os
from settings import *
from src.utils.sampu import load_model, encode, decode, sel_rand_posture, interpolate
from src.utils.visu import z_anim2dec
from src.data.post_processing import inverse_norm
import seaborn as sns
sns.set(style="darkgrid")

""" Loads the given model, randomly selects n z_means, interpolates interp_steps between them
    with slerp interpolation and decodes the output to an animation.
    The random selection can be control in sel_rand_posture function 
"""

check_model = '5'
check_epoch = '-500'
x_dataset = 'df13_50fps.csv'
z_mean_dataset = 'mean_5-500_df13_50fps'
z_sigma_dataset = 'sigma_5-500_df13_50fps'
save_stuff = True
n_postures = 4
interp_steps = 100


# Restore model to get the decoder
model = load_model(check_model, check_epoch)

# Load animation dataset and z_means
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)
df_z_mean = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_mean_dataset), index_col=0)
df_z_sigma = pd.read_csv(os.path.join(ROOT_PATH, DATA_Z_PATH, z_sigma_dataset), index_col=0)

# Latent space dimensions - 2 cols for id and category
z_dim = df_z_mean.shape[1] - 2

df_pos = sel_rand_posture(df_anim, n_postures, 'select')
anim_id, category = df_anim.loc[df_pos.index, ['id', 'category']]
start_end = df_pos.index.values.tolist()
latent_mean, latent_sigma = encode(df_pos, model)
interp = interpolate(latent_mean[0], latent_mean[1], interp_steps)

gen_anim = decode(interp, model)
output_df = pd.DataFrame(columns=joints_names, data=gen_anim)

# Inverse normalization
scaler = 'j_scaler_nao_lim_df23_50fps.pkl'
output_df = inverse_norm(output_df, scaler)
# Add the time vector
output_df['time'] = np.linspace(0.02, (output_df.shape[0] + 1) * 0.02, output_df.shape[0])
# Label for the id
output_df['id'] = 'GEN_' + anim_id
# Label for the category
output_df['category'] = category

fig = z_anim2dec(df_z_mean, interp, output_df, anim_id, category)

if save_stuff:
    fig.savefig(os.path.join(ROOT_PATH, 'reports/vae/visualizations/interpolations',
                             check_model + check_epoch + '-' + str(n_postures) + 'p-' + str(
                                 interp_steps) + 'st-' + anim_id))
    # Save generated
    output_df.to_csv(os.path.join(ROOT_PATH, 'data/generated/generated_VAE/vae_sampled/interp_npostures_anim',
                                  check_model + check_epoch + '-' + str(
                                      interp_steps) + 'st-' + anim_id + '.csv'))

