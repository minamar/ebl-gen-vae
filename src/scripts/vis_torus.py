import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


radius = 3
circles = 6  # Parallel to z axis. Equal to x2 longitudes (circle)
lat = 5
# Points on a circe (first and last are the same)


theta_t = np.linspace(0, 2*np.pi, 20)
phi_t = np.linspace(0, 2*np.pi, 9)
theta_t, phi_t = np.meshgrid(theta_t, phi_t)
R = 1.5

r = 1.5
x = (R + r * np.cos(theta_t)) * np.cos(phi_t)
y = (R + r * np.cos(theta_t)) * np.sin(phi_t)
z = r * np.sin(theta_t)

fig4 = plt.figure()
ax = plt.axes(projection='3d')

# ax1.plot_surface(x, y, z, rstride=5, cstride=5, edgecolors='k')
ax.scatter(x.flatten('C'), y.flatten('C'), z.flatten('C'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

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