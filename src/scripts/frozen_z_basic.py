import os

import pandas as pd
import seaborn as sns

from settings import *
from src.data.post_processing import inverse_norm
from src.utils.sampu import decode, load_model

sns.set(style="darkgrid")

""" Fixes two latent dimensions to 0 and varies the 3rd latent dim. 
    Range of variation -10, 10, with step 0.5
    This is to explore if there is something interested learned by each latent dim (disentanglement).
"""

check_model = '42'
check_epoch = '-200'
fr = 0.06
scaler = 'j_scaler_nao_lim_df13_50fps.pkl'
z_dim = ['l1', 'l2', 'l3']
gen_vae_dir = 'frozen_z_basic'
dest = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir)
model = load_model(check_model, check_epoch)
r = 10
step = 0.1

latent_range = np.arange(-r, r, step)

for z in z_dim:
    # Creates the latent df with z dim equal to latent range and the rest dim equal to zero.
    data = np.zeros([int(2*r/step), 3])
    df = pd.DataFrame(data, columns=z_dim)
    df[z] = latent_range

    # Decode
    a_dec = decode(df.values, model)
    df_dec = pd.DataFrame(columns=joints_names, data=a_dec)
    df_dec_norm = inverse_norm(df_dec, scaler)

    # Save
    df_dec_norm.to_csv(os.path.join(dest, z + '_var_dec.csv'))
    df.to_csv(os.path.join(dest, z + '_var_z.csv'))

