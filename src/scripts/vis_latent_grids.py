import pandas as pd
import os
from settings import *
from src.utils.visu import x_all2z, set_pub
from src.utils.sampu import load_model, encode
import matplotlib.pyplot as plt

set_pub()
fig = plt.figure()
ax = plt.axes(projection='3d')
method = 'slerp'
nsteps = 10  # per segment
fr = 0.06
radiuses = [3]  # Sampling radius [3]  #[
circles = 6  # Parallel to z axis. Equal to x2 longitudes
lat = 41  # Points on a circe (first and last are the same)
southp = int(lat / 2) + 1  # idx to split lat points to form half circle (longitude)
top_down = True  # Save postures on the longitude from north to south pole if True

# Save path
df_path = os.path.join(ROOT_PATH, DATA_SAMP, 'interp_grid_longitude/52-200/l1')

for radius in radiuses:
    phi = np.linspace(0, np.pi, circles)  # 10 times the perimeter, parallel to z axis (longitude)
    theta = np.linspace(0, 2 * np.pi, lat)  # (latidude) in parallel to x,y plane
    x = radius * np.outer(np.sin(theta), np.cos(phi))  # the plane
    y = radius * np.outer(np.sin(theta), np.sin(phi))  # the plane
    z = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis

    plt.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))

ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
# ax.axis('equal')
# ax.axis('square')
plt.show()