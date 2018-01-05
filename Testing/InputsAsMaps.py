# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:04:25 2018

@author: garrettsmith

Another, simpler approach to hanlding inputs: just turn on the relevant
features of a word at a given location in the sentence, and then let the
system gravitate after that to the nearest attractor.

For simplicity, I'll start with a 2D system. One dimension codes some feature
of a word, and the second might code the strength of a link from that word to
another. There will be a weak attractor at (0, 0) and a strong one at (1, 1).
The input will place map the system from its current location to (1, 0), and
then it will settle.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

centers = np.array([[0, 0], [1, 1]])
harmonies = np.array([0.15, 1.0])
gamma = 0.25
ndim = centers.shape[1]
noisemag = 0.01
maxsteps = 3000
tau = 0.01
x0 = [0.1, 0.1]  # starting slightly away from the attractor


# Defining phi function
def phi(x, center, gamma):
    l2norm = np.linalg.norm(x - center)
    return np.exp(-l2norm**2 / gamma)


def calc_harmony(x, centers, harmonies, gamma):
    harmony = 0
    for c in range(centers.shape[0]):
        harmony += harmonies[c] * phi(x, centers[c], gamma)
    return harmony


# A function for updating the state of the system according to the negative
# gradient of the harmony function
def step_dyn(x, centers, harmonies, gamma):
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (mult * harmonies[c]
               * (x - centers[c]) * phi(x, centers[c], gamma))
    return dx


# Running
x = np.zeros((maxsteps, ndim))
x[0, ] = x0
noise = np.random.normal(0, 1, x.shape)
H = np.zeros(maxsteps)

for t in range(0, maxsteps-1):
    if t == 100:
        x[t+1, ] = [1, 0]  # bumping out to one type of input
    elif t == 1000:
        x[t+1, ] = [0, 1]  # bumping out to another
    else:
        x[t+1, ] = (x[t, ] + tau*step_dyn(x[t, ], centers, harmonies, gamma)
                    + np.sqrt(noisemag*tau)*noise[t, ])
    H[t] = calc_harmony(x[t, ], centers, harmonies, gamma)


# Making a cool plot
delta = 0.025
x1 = np.arange(-0.1, 1.1, delta)
x2 = np.arange(-0.1, 1.1, delta)
Z = np.zeros((len(x1), len(x2)))
for i, j in product(range(len(x1)), range(len(x2))):
    Z[i, j] = calc_harmony([x1[i], x2[j]], centers, harmonies, gamma)


clines = plt.contour(x1, x2, Z, 20, cmap='RdGy_r')
plt.clabel(clines, clines.levels[::3], inline=True, fontsize=12)
plt.plot(x[:, 0], x[:, 1])
plt.show()
