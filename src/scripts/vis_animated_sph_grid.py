import pandas as pd
import os
from settings import *
from src.utils.visu import set_pub
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# Plot settings
set_pub()


radiuses = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 10]  # Sampling radius
circles = 6  # Parallel to z axis. Equal to x2 longitudes
lat = 41    # Points on a circe (first and last are the same)
southp = int(lat/2) + 1  # idx to split lat points to form half circle (longitude)
top_down = True  # Save postures on the longitude from north to south pole if True

# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')
#
# for radius in radiuses:
#     phi = np.linspace(0, np.pi, circles)     # 10 times the perimeter, parallel to z axis (longitude)
#     theta = np.linspace(0, 2 * np.pi, lat)    # (latidude) in parallel to x,y plane
#     x = radius * np.outer(np.sin(theta), np.cos(phi))   # the plane
#     y = radius * np.outer(np.sin(theta), np.sin(phi))   # the plane
#     z = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis
#
#     ax2.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))
#
# ax2.set_xlabel('LD1')
# ax2.set_ylabel('LD2')
# ax2.set_zlabel('LD3')
#
# ax2.axis('equal')
# ax2.axis('square')
#
# plt.show()


# The animated spherical grid
def update_plot(frame_number, zarray, plot):
    plot.set_data(zarray[:2, :frame_number])
    plot.set_3d_properties(zarray[2, :frame_number])

# fig = plt.figure()
# ax = plt.axes(projection='3d')

radius = 10

phi = np.linspace(0, np.pi, circles)     # 10 times the perimeter, parallel to z axis (longitude)
theta = np.linspace(0, 2 * np.pi, lat)    # (latidude) in parallel to x,y plane
x = radius * np.outer(np.sin(theta), np.cos(phi))   # the plane
y = radius * np.outer(np.sin(theta), np.sin(phi))   # the plane
z = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis

# Animated plot
zarray = np.array([x.flatten('F'), y.flatten('F'), z.flatten('F')])
plot, = ax.plot(zarray[0, 0:1], zarray[1, 0:1], zarray[2, 0:1])

# plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

ani = animation.FuncAnimation(fig, update_plot, zarray.shape[1], fargs=(zarray, plot), interval=100, blit=False)

# Setting the axes properties
ax.set_xlim([zarray[0,:].min(), zarray[0,:].max()])
ax.set_ylim([zarray[0,:].min(), zarray[0,:].max()])
ax.set_zlim([zarray[0,:].min(), zarray[0,:].max()])

ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')

ax.axis('equal')
ax.axis('square')

plt.show()