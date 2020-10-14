import json
import math
import os

import matplotlib.pyplot as plt

from settings import *
from src.utils.sampu import interp_multi, v2cat_value

""" Samples longitude lines from a horn torus grid projected on a 3D latent space.
    Decodes into the animation space with the CVAE.
"""

check_model = '63'
check_epoch = '-250'
feats_names = joints_names + leds_keys
method = 'spline'
nsteps = 20    # per segment
fr = 0.06
radiuses = [2, 3, 4, 5]#[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]  # Sampling radius
# av_list = [[0, 0], [0.5, 0.5], [1, 1], [0, 0.5], [0, 1], [0.5, 0], [1, 0], [1, 0.5], [0.5, 1]]
av_list = [[0.], [0.5], [1.]]

lat = 20    # Points on a circe (first and last are the same)
long = 9

# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'cond_torus_longitude/' + check_model + check_epoch + '_origin/l1')

for radius in radiuses:
    nsteps = math.ceil(5 * radius)
    R = radius/2
    r = radius/2
    theta_t = np.linspace(0, 2 * np.pi, lat)
    phi_t = np.linspace(0, 2 * np.pi, long)
    theta_t, phi_t = np.meshgrid(theta_t, phi_t)
    y = (R + r * np.cos(theta_t)) * np.cos(phi_t)
    z = (R + r * np.cos(theta_t)) * np.sin(phi_t)
    x = r * np.sin(theta_t)

    fig4 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))
    plt.show()
    for c in range(long - 1):

        pos_list = [list(i) for i in zip(x[c, 10:], y[c, 10:], z[c, 10:])]
        pos_list = pos_list + [list(i) for i in zip(x[c, :10], y[c, :10], z[c, :10])]
        longitude = c

        for av in av_list:
            K=None
            # Get the radians frames (dec, denorm) and the latent interpolants
            df_dec_interp, df_z_interp = interp_multi(pos_list, True, nsteps, check_model, check_epoch, method, feats_names, cond_AV=av)

            # Add 'time' column based on frequency fr
            end = df_dec_interp.shape[0] * fr + 0.02
            df_dec_interp['time'] = list(np.arange(0.02, end, fr))

            # Prepare the overview
            json_file = os.path.join(df_path, '-overview.json')

            with open(json_file, 'r') as fd:
                files_dict = json.load(fd)

            file_id = len(files_dict)
            category = v2cat_value(av[0])


            files_dict[file_id] = {
                'file_id': file_id,
                'interp_method': method,
                'interp_steps': nsteps,
                'frequency': fr,
                'model': check_model + check_epoch,
                'grid_radius': radius,
                'longitude': longitude,
                'latent_points': lat,
                'AV_score': av,
                'AV_cat': category

            }

            with open(json_file, 'w') as fd:
                fd.write(json.dumps(files_dict))

            # Save
            df_dec_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_dec_long' + str(longitude) + '_' + 'r' + str(radius) + '_' + category + '.csv'))
            df_z_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_z_long' + str(longitude) + '_' + 'r' + str(radius) + '_' + category + '.csv'))
