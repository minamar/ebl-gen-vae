import pandas as pd
import os
import re
from settings import *
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

""" Visualize the decoded joints trajectories for one latent dim varying and the rest fixed to zero. 
"""

check_model = '42'
check_epoch = '-200'
z_dim = ['l1', 'l2', 'l3']
# Directory with sampled anims
gen_vae_dir = 'frozen_z_basic'

mode = '_dec'

path_to_folder = os.path.join(ROOT_PATH, DATA_SAMP, gen_vae_dir)

# Plot settings
set_pub()
fig, ax = plt.subplots(nrows=3, ncols=6, sharex=True, figsize=(30, 18), squeeze=False)
dec_dataset = [f for f in os.listdir(path_to_folder) if mode in f]
dec_dataset.sort()

# for i in range(3):
#     df_data = pd.read_csv(os.path.join(path_to_folder, dec_dataset[i]), index_col=0)
#     df_data['range'] = np.arange(-10, 10, 0.1)
#     df_data[joints_names].plot( use_index=False, xticks=df_data['range'], ax=ax[i], legend=False)
#     df_data.describe()
# ax[2].set(xlabel="frames")
# ax[1].set(ylabel="joint angles (radians)")
# ax[1].legend(loc='center left', bbox_to_anchor=(0.96, 0.5))

df = pd.DataFrame(columns=joints_names + ['z_dim'])
for i in range(3):
    df_data = pd.read_csv(os.path.join(path_to_folder, dec_dataset[i]), index_col=0)
    df_data['z_dim'] = z_dim[i]
    df = pd.concat([df, df_data], axis=0)

df = df.pivot(columns='z_dim')
ax_dic = {0: (0, 0),  1: (0, 1),  2: (0, 2),  3: (0, 3),  4: (0, 4),
          5: (1, 0),  6: (1, 1),  7: (1, 2),  8: (1, 3),  9: (1, 4), 10: (1, 5),
          11: (2, 0), 12: (2, 1), 13: (2, 2), 14: (2, 3), 15: (2, 4), 16: (2, 5)}
for i, j in zip(range(17), joints_names):
    dfu = df[j]
    dfu.plot(title=joints_names[i], xticks=None, xlim=[0,200], ax=ax[ax_dic[i]], legend=False)


# ax[9].legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
fig.delaxes(ax[0, 5])
plt.tight_layout()
ax[0, 4].legend(loc='center left', bbox_to_anchor=(1, 0.5), handlelength=5, borderpad=1.2, labelspacing=1.2)
ax[2, 0].set_xticklabels([-10, -5, 0, 5, 10])
plt.ylabel('joint angles (radians)"')


plt.show()

d = df['HeadPitch']
sig = d['l1']

from numpy.fft import rfft
from numpy import argmax
from scipy.signal import blackmanharris



def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)


    return (xv, yv)

def freq_from_HPS(sig, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    """
    windowed = sig * blackmanharris(len(sig))

    from pylab import subplot, plot, log, copy, show

    # harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms, 1, 1)
    plot(log(c))
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)
        print('Pass %d: %f Hz' % (x, fs * true_i / len(windowed)))
        c *= a
        subplot(maxharms, 1, x)
        plot(log(c))
    show()


