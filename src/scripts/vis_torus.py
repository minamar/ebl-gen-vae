import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


theta_t = np.linspace(0, 2*np.pi, 50)
phi_t = np.linspace(0, 2*np.pi, 9)
theta_t, phi_t = np.meshgrid(theta_t, phi_t)
R = 1.5

r = 1.5
x = (R + r * np.cos(theta_t)) * np.cos(phi_t)
y = (R + r * np.cos(theta_t)) * np.sin(phi_t)
z = r * np.sin(theta_t)

fig4 = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, alpha=0.10, edgecolor='black')

for i in range(8):
    ax.plot(x[i, :].flatten('C'), y[i, :].flatten('C'), z[i, :].flatten('C'))

# ax.scatter(x[4, 0:3], y[4, 0:3], z[4, 0:3], c='red')
# ax.scatter(x, y, z, c="grey")
# ax.scatter(x.flatten('C')[0], y.flatten('C')[0], z.flatten('C')[0], c='blue', marker='*', s=4)
# ax.scatter(x.flatten('C')[1], y.flatten('C')[1], z.flatten('C')[1], c='blue', marker='*', s=4)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

#
# radius = 3
# circles = 6  # Parallel to z axis. Equal to x2 longitudes (circle)
# lat = 41    # Points on a circe (first and last are the same)
#
# phi = np.linspace(0, np.pi, circles)  # 10 times the perimeter, parallel to z axis (longitude)
# theta = np.linspace(0, 2 * np.pi, lat)  # (latidude) in parallel to x,y plane
# x = radius * np.outer(np.sin(theta), np.cos(phi))  # the plane
# y = radius * np.outer(np.sin(theta), np.sin(phi))  # the plane
# z = radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis
# ax.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))

# radius = 0.5
# phi = np.linspace(0, np.pi, circles)  # 10 times the perimeter, parallel to z axis (longitude)
# theta = np.linspace(0, 2 * np.pi, lat)  # (latidude) in parallel to x,y plane
# x = 1 + radius * np.outer(np.sin(theta), np.cos(phi))  # the plane
# y = 1 + radius * np.outer(np.sin(theta), np.sin(phi))  # the plane
# z = 1 + radius * np.outer(np.cos(theta), np.ones_like(phi))  # the axis
# ax.plot(x.flatten('F'), y.flatten('F'), z.flatten('F'))

plt.show()
n= 1