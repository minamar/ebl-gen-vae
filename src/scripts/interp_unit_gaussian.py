import pandas as pd
import os
import json
from settings import *
from src.utils.sampu import interp_multi, sel_pos_frame
import seaborn as sns
sns.set(style="darkgrid")

""" Sample a 3d unit gaussian, directly interpolate them in the latent space, decode """

check_model = '42'
check_epoch = '-200'
methods = ['slerp']  # slerp, lerp, spline
nsteps = 40    # per segment
fr = 0.06
n_pos = 6
std = 3  # Sampling radius


# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_unit_gaussian')

# Sample a 3D diag unit gaussian

mean = [0, 0, 0]
cov = [[std, 0, 0], [0, std, 0], [0, 0, std]]
pos_list = np.random.multivariate_normal(mean, cov, n_pos)
pos_list = pos_list.tolist()


# =======
# import numpy as np
#
# def sample_spherical(npoints, ndim=3):
#     vec = np.random.randn(ndim, npoints)
#     vec /= np.linalg.norm(vec, axis=0)
#     return vec
#
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# r = 6
# phi = np.linspace(0, np.pi, 10)
# theta = np.linspace(0, 2 * np.pi, 20)
# x = r * np.outer(np.sin(theta), np.cos(phi))
# y = r * np.outer(np.sin(theta), np.sin(phi))
# z = r * np.outer(np.cos(theta), np.ones_like(phi))
#
# xi, yi, zi = sample_spherical(10)
#
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
# # ax.scatter(x, y, z)
# ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
# # ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# plt.show()
# pos_list = [list(i) for i in zip(x.flatten('F'),y.flatten('F'),z.flatten('F'))]
# ==========

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

