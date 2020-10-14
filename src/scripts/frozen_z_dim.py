import os

import pandas as pd
import seaborn as sns

from settings import *
from src.data.post_processing import inverse_norm
from src.utils.sampu import decode, load_model

sns.set(style="darkgrid")

""" Using 5 random samples from a 3d unit gaussian, 2 dims are fixed with the latent standInit and the 3rd dim varies 
    according to the interpolant. This is to explore if there is something interested learned by each latent dim
    (disentanglement).
"""

check_model = '42'
check_epoch = '-200'
method = 'slerp'  # slerp, lerp, bspline
nsteps = 100    # per segment
fr = 0.06
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
data= '6_z_lerp.csv'
gen_vae_dir = 'frozen_z_dim'
df = pd.read_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data), index_col=0)

df_l1_var = df.iloc[:, :-1]
df_l1_var['l2'] = df.loc[0, 'l2']
df_l1_var['l3'] = df.loc[0, 'l3']

df_l2_var = df.iloc[:, :-1]
df_l2_var['l1'] = df.loc[0, 'l1']
df_l2_var['l3'] = df.loc[0, 'l3']

df_l3_var = df.iloc[:, :-1]
df_l3_var['l1'] = df.loc[0, 'l1']
df_l3_var['l2'] = df.loc[0, 'l2']

# Decode
model = load_model(check_model, check_epoch)
l1_var_dec = decode(df_l1_var.values, model)
df_dec_l1_var = pd.DataFrame(columns=joints_names, data=l1_var_dec)
df_dec_l1_var_norm = inverse_norm(df_dec_l1_var, scaler)

l2_var_dec = decode(df_l2_var.values, model)
df_dec_l2_var = pd.DataFrame(columns=joints_names, data=l2_var_dec)
df_dec_l2_var_norm = inverse_norm(df_dec_l2_var, scaler)

l3_var_dec = decode(df_l3_var.values, model)
df_dec_l3_var = pd.DataFrame(columns=joints_names, data=l3_var_dec)
df_dec_l3_var_norm = inverse_norm(df_dec_l3_var, scaler)

# Save
df_dec_l1_var_norm.to_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data.split('.')[0] + '_l1_var.csv'))
df_dec_l2_var_norm.to_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data.split('.')[0] + '_l2_var.csv'))
df_dec_l3_var_norm.to_csv(os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir, data.split('.')[0] + '_l3_var.csv'))

