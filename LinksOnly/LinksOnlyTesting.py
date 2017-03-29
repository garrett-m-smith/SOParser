# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:40 2017

@author: garrettsmith
"""

import numpy as np
#import matplotlib.pyplot as plt

#from scipy.integrate import odeint

nlinks = 6
# WTA weights, Whit
#W = np.array([[1., 2, 0, 0, 0, 2], 
#              [2, 1, 0, 0, 2, 0],
#              [0, 0, 1, 2, 0, 0],
#              [0, 0, 2, 1, 0, 0],
#              [0, 2, 0, 0, 1, 2],
#              [2, 0, 0, 0, 2, 1]])

# My original idea
W = np.array([[1., 2, 0, 0, 0, 2], 
              [2, 1, 2, 0, 2, 0],
              [0, 2, 1, 2, 0, 0],
              [0, 0, 2, 1, 2, 0],
              [0, 2, 0, 2, 1, 2],
              [2, 0, 0, 0, 2, 1]])

link_names = ['N1->N(V)', 'N1->N(P)', 'P->P(N1)', 'P->Det(N2)', 
'N2->N(P}', 'N2->N(V)']

dx = np.zeros(nlinks)
def dyn(x, t):
    dx = x * (1 - W @ x)
    return dx

#x0 = np.random.uniform(0.1, 0.2, nlinks)
#x0 = np.array([0.05, 0.5, 0.15, 0.45, 0.1, 0.55])

#noise = np.random.normal(0, tau, xhist.shape)

#xhist = odeint(dyn, x0, tvec)

# Individual runs
#for t in range(1, len(tvec)):
#    xhist[t,:] = (xhist[t-1,] + tau * (xhist[t-1,] 
#    * (1 - W @ xhist[t-1,]) + noise[t,:]))
#    
#    if t == 30:
#        # Turn boost P-P(N1) = intro Prep
#        xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
#        # Also turn on its competitor: N1-N(P)
#        xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
#    if t == 60:
#        # Intro N2-related links: P-Det(N2)
#        xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
#        # N2-N(P)
#        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
#        # N2-N(V)
#        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
#
#
#plt.figure()
#plt.ylim(-0.1, 1.1)
#for d in range(xhist.shape[1]):
#    plt.plot(xhist[:,d], label = link_names[d])
#    xpos = d * (len(xhist[:,d]) / len(link_names))
#    ypos = xhist[xpos, d]
#    plt.text(xpos, ypos, link_names[d])
#plt.legend()
#plt.show()
#
#for d in range(len(link_names)):
#    print('{}:\t{}'.format(link_names[d], np.round(xhist[-1,d], 4)))

# Monte Carlo
nrep = 1000
n1headed = 0
n2headed = 0
other = 0
noisemag = 1.
finals = np.zeros((nrep, nlinks))
adj = 2.

x0 = np.array([0.1] * nlinks)

# Setting first word to between its current state and 1
x0[0] = x0[0] + (1 - x0[0]) / adj

tau = 0.01
nsec = 40
tvec = np.linspace(0, nsec, nsec/tau + 1)
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0


for rep in range(nrep):
#    x0 = np.random.uniform(0.0, 1.0, nlinks)
#    xx = odeint(dyn, x0, tvec)
    xhist = np.zeros((len(tvec), nlinks))
    xhist[0,] = x0
    
    noise = np.random.normal(0, noisemag, xhist.shape)
    for t in range(1, len(tvec)):
        xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
        * (1 - W @ xhist[t-1,]) + noise[t,:]), -0.1, 1.1)
    
        if t == 5:
            # Turn boost P-P(N1) = intro Prep
            xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
            # Also turn on its competitor: N1-N(P)
            xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
        if t == 10:
            # Intro N2-related links: P-Det(N2)
            xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
            # N2-N(P)
            xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
            # N2-N(V)
            xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
    
    final = np.round(xhist[-1,], 2)
    if np.allclose(final, np.array([1., 0, 1, 0, 1, 0]), rtol = 0.1, atol = 0.1):
        n1headed += 1
    elif np.allclose(final, np.array([0., 1, 0, 1, 0, 1]), rtol = 0.1, atol = 0.1):
        n2headed += 1
    else:
        other += 1
    finals[rep,:] = final

print('N1 headed: {0}\nN2 headed: {1}\nOther: {2}'.format(n1headed, 
      n2headed, other))


