# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:56:09 2017

@author: garrettsmith

Testing a new way of handling input

I started with the Smolensky/Cho approach, which just adds the inputs to the
dynamics (dx/dt = del H + input), but this can't pull the system towards 0 if
it is already non-zero.

So, I think the easiest way to go is to have the input be just another RBF. 
Here, I've weighted the input RBF so that it has a large enough impact to
overcome the harmony dip between the two attractors.

To do:
    --Explore the effect of weighting the input RBF; this is another free
    parameter.
    --Explore how long the input should be turned on to have an effect
    --Implement a way of having the input affect only certain dimensions,
    while leaving the other dimensions untouched.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Just trying out a 1D version with two attractors

centers = np.array([0, 1])
harmonies = np.ones(2)
gamma = 0.25
ndim = 1
noisemag = 0.01
maxsteps = 2000
tau = 0.01
x0 = 0.4

# Defining phi function
def phi(x, center, gamma):
    l2norm = np.linalg.norm(x - center)
    return np.exp(-l2norm**2 / gamma)


# A function for updating the state of the system according to the negative
# gradient of the harmony function
def step_dyn(x, centers, harmonies, gamma, ipt=None):
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (mult * harmonies[c]
               * (x - centers[c]) * phi(x, centers[c], gamma))
    #return dx + ipt
    if ipt is not None:
        return dx + 2*mult*(x-ipt)*phi(x, ipt, gamma)
    else:
        return dx


# Running
x = np.zeros(maxsteps)
x[0] = x0
noise = np.random.normal(0, 1, x.shape)
H = np.zeros(maxsteps)
for t in range(0, maxsteps-1):
    if t < 250:
        ipt = None
    elif t >= 250 and t < 750:
        ipt = 1.0
    else:
        ipt = 0.
    x[t+1] = (x[t] + tau*step_dyn(x[t], centers, harmonies, gamma, ipt)
              + np.sqrt(noisemag*tau)*noise[t])
    H[t] = phi(x[t], centers[0], gamma)+phi(x[t], centers[1], gamma)


#plt.plot(x)
plt.scatter(np.arange(0, maxsteps), x, c=cm.coolwarm(H),
            edgecolors=None)
#plt.colorbar(cax=cm.coolwarm(H/max(H)))
plt.show()
