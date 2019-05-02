import os
import json
from settings import *
from src.utils.sampu import interp_multi

""" Sample longitude lines from the spherical grid. """

check_model = '53'
check_epoch = '-500'
feats_names = joints_names + leds_keys
method = 'slerp'
nsteps = 10    # per segment
fr = 0.06
radiuses = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 10]  # Sampling radius
circles = 6  # Parallel to z axis. Equal to x2 longitudes
lat = 41    # Points on a circe (first and last are the same)
southp = int(lat/2) + 1  # idx to split lat points to form half circle (longitude)
top_down = True  # Save postures on the longitude from north to south pole if True

# TODO: add param for the model id
# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_grid_longitude/53-500/l3')

for radius in radiuses:
    phi = np.linspace(0, np.pi, circles)     # 10 times the perimeter, parallel to z axis (longitude)
    theta = np.linspace(0, 2 * np.pi, lat)    # (latidude) in parallel to x,y plane
    y = radius * np.outer(np.sin(theta), np.cos(phi))   # the plane
    z = radius * np.outer(np.sin(theta), np.sin(phi))   # the plane
    x = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis

    for hem in range(2):

        for c in range(circles - 1):

            if hem == 0:
                pos_list = [list(i) for i in zip(x[:southp, c], y[:southp, c], z[:southp, c])]
                long = c
            else:
                pos_list = [list(i) for i in zip(x[(southp - 1):, c], y[(southp - 1):, c], z[(southp - 1):, c])]
                pos_list = pos_list[::-1]
                long = c + circles - 1

            # Get the radians frames (dec, denorm) and the latent interpolants
            df_dec_interp, df_z_interp = interp_multi(pos_list, True, nsteps, check_model, check_epoch, method, feats_names)

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
                'longitude': long,
                'latent_points': long * lat
            }

            with open(json_file, 'w') as fd:
                fd.write(json.dumps(files_dict))

            # Save
            df_dec_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_dec_long' + str(long) + '_' + 'r' + str(radius) + '.csv'))
            df_z_interp.to_csv(
                os.path.join(df_path, str(file_id) + '_z_long' + str(long) + '_' + 'r' + str(radius) + '.csv'))
