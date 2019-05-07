import pandas as pd
import os
import json
from settings import *
from src.utils.sampu import interp_multi, sel_pos_frame
import seaborn as sns
sns.set(style="darkgrid")

""" Sample a 3d unit gaussian, directly interpolate them in the latent space, decode """

check_model = '61'
check_epoch = '-100'
latent_dim = 2
method = 'slerp'  # slerp, lerp, spline
nsteps = 30    # interpolation steps per segment
fr = 0.06
n_pos = 5   # Key postures
std = 1  # Sampling radius
feats_names = joints_names + leds_keys
va_list = [[0, 0], [0.5, 0.5], [1, 1], [0, 0.5], [0, 1], [0.5, 0], [1, 0], [1, 0.5], [0.5, 1]]

gen_vae_dir = 'cond_gaussian_sampling/' + check_model + check_epoch
# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir)

# Sample a 3D diag unit gaussian

mean = [0.] * latent_dim
cov = np.diag([std] * latent_dim)
pos_list = np.random.multivariate_normal(mean, cov, n_pos)
pos_list = pos_list.tolist()
pos_list = [[0.01] * latent_dim] + pos_list + [[0.01] * latent_dim]


for va in va_list:
    # Get the radians frames (dec, denorm) and the latent interpolants
    df_dec_interp, df_z_interp = interp_multi(pos_list, True, nsteps, check_model, check_epoch, method, feats_names, cond_VA=va)

    # Add 'time' column based on frequency fr
    end = df_dec_interp.shape[0] * fr + 0.02
    df_dec_interp['time'] = list(np.arange(0.02, end, fr))

    # Prepare the overview
    json_file = os.path.join(df_path, '-overview.json')

    with open(json_file, 'r') as fd:
        files_dict = json.load(fd)

    file_id = len(files_dict)

    files_dict[file_id] = {
        'file_id': file_id,
        'interp_method': method,
        'interp_steps': nsteps,
        'frequency': fr,
        'model': check_model + check_epoch,
        'sampling radius': std,
        'n_postures': n_pos,
        'VA_score': va,
        'VA_cat': va2cat[str(va)]
    }

    with open(json_file, 'w') as fd:
        fd.write(json.dumps(files_dict))

    # Save
    df_dec_interp.to_csv(os.path.join(df_path, str(file_id) + '_dec_' + 'r' + str(std) + '_' +  va2cat[str(va)] + '.csv'))
    df_z_interp.to_csv(os.path.join(df_path, str(file_id) + '_z_' + 'r' + str(std) + '_' +  va2cat[str(va)] + '.csv'))

