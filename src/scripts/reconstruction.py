import pandas as pd
import os
from settings import *
from src.utils.sampu import load_model, encode, decode
from src.data.post_processing import inverse_norm

""" Trivially encodes a set of animations and then decodes them to inspect the quality of the reconstruction.
"""

check_model = '54'
check_epoch = '-500'
x_dataset = 'df32_25fps.csv'
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'


# Restore model to get the decoder
model = load_model(check_model, check_epoch)

# Load animation dataset and z_means
df_anim = pd.read_csv(os.path.join(ROOT_PATH, DATA_X_PATH, x_dataset), index_col=0)

df_anim = df_anim[~df_anim.id.str.contains('_tr')]
x = df_anim.drop(columns=['time', 'id', 'category'], inplace=False)

# Encode the animation set x
latent_mean, latent_sigma = encode(x, model)

# Reconstruct animations from their trained latent_mean
x_reconstruction = decode(latent_mean, model)

output_df = pd.DataFrame(columns=joints_names, data=x_reconstruction)

# Inverse normalization
output_df = inverse_norm(output_df, scaler)

output_df['time'] = df_anim['time']
output_df['category'] = df_anim['category']
output_df['id'] = df_anim['id']

dest = os.path.join(ROOT_PATH, DATA_RECO, x_dataset.split('.')[0])

if not os.path.exists(dest):
    os.makedirs(dest)

dest = os.path.join(dest, check_model + check_epoch)

if not os.path.exists(dest):
    os.makedirs(dest)

# All the reconstructed animations in one df
output_df.to_csv(os.path.join(dest, check_model + check_epoch + '_all.csv'))

# Each reconstructed in individual df
ids = df_anim['id'].unique()
for id_anim in ids:
    anim_output_df = output_df.loc[output_df['id'] == id_anim, :]
    anim_output_df.reset_index(inplace=True, drop=True)
    anim_output_df.to_csv(os.path.join(dest, 'REC_' + id_anim + '.csv'))


