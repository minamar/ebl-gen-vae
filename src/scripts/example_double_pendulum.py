# comment these lines if you want interactive mode,
# i.e., if you want to see the animation in real time.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import ode


def equations(t, y, args):
    """ the equations for the double pendulum """
    x1 = y[0] # x1 = theta1, angle
    x2 = y[1] # x2 = theta2, angle
    p1 = y[2] # p1 = omega1, angular velocity
    p2 = y[3] # p2 = omega2, angular velocity
    l1,l2,m1,m2,g = args
    x1_eq = p1
    x2_eq = p2
    p1_eq = -((g*(2*m1+m2)*np.sin(x1)+m2*(g*np.sin(x1-2*x2)+2*(l2*p2**2+l1*p1**2*np.cos(x1-x2))*np.sin(x1-x2)))/(2*l1*(m1+m2-m2*(np.cos(x1-x2))**2)))
    p2_eq = ((l1*(m1+m2)*p1**2+g*(m1+m2)*np.cos(x1)+l2*m2*p2**2*np.cos(x1-x2))*np.sin(x1-x2))/(l2*(m1+m2-m2*(np.cos(x1-x2))**2))
    return [x1_eq, x2_eq, p1_eq, p2_eq]


def calculate_trajectory(args,time,y0):
    """ uses scipy's ode itegrator to simulate the equations """
    t0,t1,dt = time
    r = ode(equations).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(args)
    data=[[t0, y0[0], y0[1], y0[2], y0[3] ]]
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        data.append([r.t, r.y[0], r.y[1], r.y[2], r.y[3] ])
    return np.array(data)


def from_angle_to_xy(args,angles):
    """ converts angles into xy positions """
    l1,l2,m1,m2,g = args
    time,theta1,theta2 = angles.T
    x1 =  l1*np.sin(theta1)
    y1 = -l1*np.cos(theta1)
    x2 =  l2*np.sin(theta2) + x1
    y2 = -l2*np.cos(theta2) + y1
    return np.array([time,x1,y1,x2,y2]).T

l1 = 0.5 # length of arms
l2 = 0.5
m1 = 1.0 # mass of the pendulum
m2 = 1.0
g  = 10.0 # acceleration of gravity
args = [l1,l2,m1,m2,g]
fps = 80
total_time = 5 # seconds
time = [0.0,total_time,1.0/fps] # start, finish, dt
ic   = [np.pi*0.65, np.pi*1.1, 0.0, 0.0]

# here the magic happens
d = calculate_trajectory(args,time,ic)
data_TXY = from_angle_to_xy(args,d[:,:3])

