import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


# 3D example
total_rad = 10
z_factor = 3

num_sample_pts = 2
s_sample = np.linspace(0, total_rad, num_sample_pts)
x_sample = np.array([-0.5402, 1.2958])
y_sample = np.array([-0.3090, -1.0266])
z_sample = np.array([-0.1536, 1.9205])

num_true_pts = 100
tck, u = interpolate.splprep([x_sample,y_sample,z_sample], k=1, s=2)
x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
u_fine = np.linspace(0,1,num_true_pts)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
ax3d.plot(x_sample, y_sample, z_sample, 'r*')
ax3d.plot(x_knots, y_knots, z_knots, 'go')
ax3d.plot(x_fine, y_fine, z_fine, 'g')
fig2.show()
plt.show()


from splipy import *
from splipy.utils import smooth

def plot_3D_curve(curve, show_controlpoints=False):
    t = np.linspace(curve.start(), curve.end(), 150)
    x = curve(t)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:,0], x[:,1], x[:,2])
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    if show_controlpoints:
        plt.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'rs-')
    plt.show()


pos = [[0., 0., 0], [1., 1., 1.], [0.2, 0.2, 0.2]]

dist = np.linalg.norm(np.array(pos[0])-np.array(pos[1])) / 3.
l = curve_factory.line(pos[0], pos[1])
l.raise_order(2)
plot_3D_curve(l, show_controlpoints=True)

pts = l.controlpoints
pts[1,1] = pts[1,1] - dist
pts[2, 1] = pts[2, 1] + dist
pts[1, 2] = pts[1, 2] + dist
pts[2, 2] = pts[2, 2] - dist

c = curve_factory.bezier(pts)

dist2 = np.linalg.norm(np.array(pos[1])-np.array(pos[2])) / 3.
l2 = curve_factory.line(pos[1], pos[2])
l2.raise_order(2)
plot_3D_curve(l2, show_controlpoints=True)

pts2 = l2.controlpoints
pts2[1,1] = pts2[1,1] - dist2
pts2[2, 1] = pts2[2, 1] + dist2
pts2[1, 2] = pts2[1, 2] + dist2
pts2[2, 2] = pts2[2, 2] - dist2

c2 = curve_factory.bezier(pts2)

pts_comb = pts.tolist()[0:3] + pts2.tolist()
# pts_comb = [pts.tolist()[0:3], pts2.tolist()]

# Cubic bezier needs two extra control points in between start and stop
c_comb = curve_factory.bezier(pts_comb)
plot_3D_curve(c, show_controlpoints=False)
