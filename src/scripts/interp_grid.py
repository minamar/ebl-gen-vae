import os
import json
from settings import *
from src.utils.sampu import interp_multi
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

""" Sample from grid """

check_model = '42'
check_epoch = '-200'
method = 'slerp'
nsteps = 5    # per segment
fr = 0.06
radiuses = [0.5, 1., 1.5, 2, 3, 4, 5, 7, 10, 20, 50, 100]  # Sampling radius
long = 10
lat = 20
# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_grid')


for radius in radiuses:
    phi = np.linspace(0, np.pi, long)     # 10 times the perimeter, parallel to z axis (longitude)
    theta = np.linspace(0, 2 * np.pi, lat)    # (latidude) in parallel to x,y plane
    x = radius * np.outer(np.sin(theta), np.cos(phi))
    y = radius * np.outer(np.sin(theta), np.sin(phi))
    z = radius * np.outer(np.cos(theta), np.ones_like(phi))  #

    # fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    # ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    # ax.scatter(x[:, 0], y[:, 0], z[:, 0], s=100, c='b', zorder=10)
    # ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    # Flatten 'F' will run around the perimeter in parallel to z axis 10 times
    pos_list = [list(i) for i in zip(x.flatten('F'), y.flatten('F'), z.flatten('F'))]

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
        'grid_radius': radius,
        'latent_points': long * lat
    }

    with open(json_file, 'w') as fd:
        fd.write(json.dumps(files_dict))

    # Save
    df_dec_interp.to_csv(os.path.join(df_path, str(file_id) + '_dec_' + method + '.csv'))
    df_z_interp.to_csv(os.path.join(df_path, str(file_id) + '_z_' + method + '.csv'))

