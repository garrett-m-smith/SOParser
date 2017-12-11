# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 08:43:49 2017

@author: garrettsmith

Trying out the Lyapunov approach to the dynamics, following:
    Han et al. (1989), Ciocoiu (1996, 2009), Muezzinoglu & Zurada (2006)

Starting with three fixed points: the origin (before two words have been
introduced), (1, 0) and (0, 1), representing two possible attachments between 
two words.

The two attractors should have different feature matches, affecting the
probability of settling at each one

Phenomena to keep track of:
    --Soft deflection
    --Terminal attraction
    --Saddles masquerading as attractors
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# Defining functions
def l2norm(x):
    return np.sqrt(x @ x)


def rbf(x, c, gamma):
    phi = np.exp(-l2norm(x-c)**2/gamma)
    return phi


def dyn(x, c, gamma, weights, ipt):
    """Takes the current position, a list of attractors, and the gamma
    parameters as arguments and updates the state.
    """
    dx = (weights[0]*(-2./gamma) * (x - c[0]) * rbf(x, c[0], gamma)
          + weights[1]*(-2./gamma) * (x - c[1]) * rbf(x, c[1], gamma)
          + weights[2]*(-2./gamma) * (x - c[2]) * rbf(x, c[2], gamma) + ipt*x)
    return dx


def energy(x, c, gamma, weights):
    e = 0
    for i in range(len(c)):
        e += weights[i]*rbf(x, c[i], gamma)
    return e


def speed(x, c, gamma, weights):
    vel = dyn(x, c, gamma, weights, [0, 0])
    return l2norm(vel)


# Setting up the basics for the simulation
ndim = 2
tlen = 300
c0 = np.array([0., 0])
c1 = np.array([1., 0])
c2 = np.array([0., 1])
cs = [c0, c1, c2]
# Feature matches/size of basins
weights = np.array([1, 2, 3.])
gamma = 0.25
x = np.zeros((tlen, ndim))
# Trying out init. cond. a little away from attr. at origin
x[0, ] = [0.5]*ndim
tau = 0.01
e = np.zeros(tlen)
noise = np.sqrt(tau) * np.random.normal(0, 0.15, (tlen, ndim))
speeds = np.zeros(tlen)
ipt = np.array([0., 0])
t = 1
trigger = 0

while t < tlen:  # and not speed(x[t, ], cs, gamma, weights) < 0.2:
    x[t] = x[t-1, ]+(tau
                     * dyn(x[t-1, ], cs, gamma, weights, ipt) + noise[t-1, ])
    e[t-1, ] = energy(x[t-1, ], cs, gamma, weights)
    speeds[t-1, ] = speed(x[t-1, ], cs, gamma, weights)
    if speeds[t-1, ] < 0.2 and trigger == 0:
        #ipt = 3.5*(c1 - x[t-1, ])
        ipt = [8, -7.]
        trigger = 1
    if speeds[t-1, ] < 0.1 and trigger == 1 and t > 100:
        ipt = [0., 0]
        break
    t += 1

plt.plot(x)
#plt.ylim(-0.1, 1.1)
plt.plot(np.trim_zeros(e)/np.max(np.trim_zeros(e)), alpha=0.5)
plt.plot(np.trim_zeros(speeds) / np.max(np.trim_zeros(speeds)), alpha=0.5)
plt.legend(['x1', 'x2', 'Energy', 'Speed'])
plt.show()

# Plotting energy contours
#plt.figure(figsize=(5,5))
#x1 = np.linspace(-0.2, 1.2)
#x2 = np.linspace(-0.2, 1.2)
#E = np.zeros((len(x1), len(x2)))
#for i, j in product(range(len(x1)), range(len(x2))):
#    E[i, j] = energy([x1[i], x2[j]], cs, gamma, weights)
#cplot = plt.contour(x1, x2, E, 10)
#plt.clabel(cplot, inline=1)
#plt.show()
