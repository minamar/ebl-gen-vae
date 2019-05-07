import os
import json
from settings import *
from src.utils.sampu import interp_multi

""" Sample longitude lines from the spherical grid. Only for 3D latent space """

check_model = '60'
check_epoch = '-100'
feats_names = joints_names + leds_keys
method = 'slerp'
nsteps = 10    # per segment
fr = 0.06
radiuses = [0.5, 1, 1.5, 2, 2.5, 3]  # Sampling radius
va_list = [[0, 0], [0.5, 0.5], [1, 1], [0, 0.5], [0, 1], [0.5, 0], [1, 0], [1, 0.5], [0.5, 1]]
lat = 20    # Points on a circe (first and last are the same)
long = 9

# TODO: add param for the model id
# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'cond_torus_longitude/60-100')

for radius in radiuses:
    R = radius/2
    r = radius/2
    theta_t = np.linspace(0, 2 * np.pi, lat)
    phi_t = np.linspace(0, 2 * np.pi, long)
    theta_t, phi_t = np.meshgrid(theta_t, phi_t)
    x = (R + r * np.cos(theta_t)) * np.cos(phi_t)
    y = (R + r * np.cos(theta_t)) * np.sin(phi_t)
    z = r * np.sin(theta_t)

    for c in range(long - 1):

        pos_list = [list(i) for i in zip(x[c, :], y[c, :], z[c, :])]
        longitude = c

        for va in va_list:
            K=None
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
                'grid_radius': radius,
                'longitude': longitude,
                'latent_points': lat,
                'VA_score': va,
                'VA_cat': va2cat[str(va)]

            }

            with open(json_file, 'w') as fd:
                fd.write(json.dumps(files_dict))

            # Save
            df_dec_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_dec_long' + str(longitude) + '_' + 'r' + str(radius) + '_' +  va2cat[str(va)] + '.csv'))
            df_z_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_z_long' + str(longitude) + '_' + 'r' + str(radius) + '_' +  va2cat[str(va)] + '.csv'))
