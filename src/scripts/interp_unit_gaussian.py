import pandas as pd
import os
import json
from settings import *
from src.utils.sampu import interp_multi, sel_pos_frame
import seaborn as sns
sns.set(style="darkgrid")

""" Sample a 3d unit gaussian, directly interpolate them in the latent space, decode """

check_model = '52'
check_epoch = '-200'
methods = ['spline']  # slerp, lerp, spline
nsteps = 100    # interpolation steps per segment
fr = 0.06
n_pos = 5   # Key postures
std = 5  # Sampling radius


# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_unit_gaussian')

# Sample a 3D diag unit gaussian

mean = [0, 0, 0]
cov = [[std, 0, 0], [0, std, 0], [0, 0, std]]
pos_list = np.random.multivariate_normal(mean, cov, n_pos)
pos_list = pos_list.tolist()
pos_list = [[0, 0, 0]] + pos_list + [[0, 0, 0]]


for method in methods:
    # Get the radians frames (dec, denorm) and the latent interpolants
    df_dec_interp, df_z_interp = interp_multi(pos_list, True, nsteps, check_model, check_epoch, method)

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
        'sampling radius': std
    }

    with open(json_file, 'w') as fd:
        fd.write(json.dumps(files_dict))

    # Save
    df_dec_interp.to_csv(os.path.join(df_path, str(file_id) + '_dec_' + method + '.csv'))
    df_z_interp.to_csv(os.path.join(df_path, str(file_id) + '_z_' + method + '.csv'))

