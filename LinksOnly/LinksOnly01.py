# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith

Introducing feature vectors
"""

import numpy as np
import matplotlib.pyplot as plt


nlinks = 6
link_names = ['N1->N(V)', 'N1->N(P)', 'P->P(N1)', 'P->Det(N2)', 
'N2->N(P)', 'N2->N(V)']

# Possibly add number features?
# Features: +-Container, +-PhysConfig +-AbstractN2, +-N, +-OtherPOS
# Head attachment sites
nfeat = 5
N_V = np.array([1., 1, 0, 1, 0])
N_P = np.array([1., 0.5, 0.5, 1, 0])
P_N1 = np.array([0., 0, 0, 0, 1])
Det_N2 = np.array([0., 0, 0, 0, 1])

# Sending attachment sites
box = np.array([1, 1, 0, 1, 0])
group = np.array([1, 0, 0, 1, 0])
lot = np.array([0, 0, 1, 1, 0])
N2 = np.array([1, 1, 0, 1, 0])
of = np.array([0., 0, 0, 0, 1])

box_N_V = box @ N_V / nfeat
box_N_P = box @ N_P / nfeat
group_N_V = group @ N_V / nfeat
group_N_P = group @ N_P / nfeat
lot_N_V = lot @ N_V / nfeat
lot_N_P = lot @ N_P / nfeat
of_P_N1 = of @ P_N1 / nfeat
of_Det_N2 = of @ Det_N2 / nfeat
N2_N_P = N2 @ N_P / nfeat
N2_N_V = N2 @ N_V / nfeat

# box of N2s
box_of_N2 = np.array([box_N_V, box_N_P, of_P_N1, of_Det_N2, N2_N_P, N2_N_V])

# WTA weights, Whit
#W = np.array([[1., 2, 0, 0, 0, 2], 
#              [2, 1, 0, 0, 2, 0],
#              [0, 0, 1, 2, 0, 0],
#              [0, 0, 2, 1, 0, 0],
#              [0, 2, 0, 0, 1, 2],
#              [2, 0, 0, 0, 2, 1]])
# My original idea
# Seems to work. Instead of saying a treelet changes its state to make certain
# attachments change their features, we just say, either one attachment can
# win, or another can, but not both. Same effect, less machinery
W = np.array([[1., 2, 0, 0, 0, 2], 
              [2, 1, 2, 0, 2, 0],
              [0, 2, 1, 2, 0, 0],
              [0, 0, 2, 1, 2, 0],
              [0, 2, 0, 2, 1, 2],
              [2, 0, 0, 0, 2, 1]])

#x0 = np.random.uniform(0.1, 0.2, nlinks)
x0 = np.array([0.1] * nlinks)

tau = 0.1
nsec = 100
tvec = np.linspace(0, nsec, nsec/tau + 1)
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0
noisemag = 0.1
noise = np.random.normal(0, noisemag, xhist.shape)
adj = 3.


# Individual runs
for t in range(1, len(tvec)):
#    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#    * (1 - W @ xhist[t-1,]) + noise[t,:]), -0.1, 1.1)
    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
    * (box_of_N2 - W @ (box_of_N2 * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
    
    if t == 30:
        # Turn boost P-P(N1) = intro Prep
        xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
        # Also turn on its competitor: N1-N(P)
        xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
    if t == 60:
        # Intro N2-related links: P-Det(N2)
        xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
        # N2-N(P)
        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
        # N2-N(V)
        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj


plt.figure()
plt.ylim(-0.1, 1.1)
for d in range(xhist.shape[1]):
    plt.plot(xhist[:,d], label = link_names[d])
    xpos = d * (len(xhist[:,d]) / len(link_names))
    ypos = xhist[xpos, d]
    plt.text(xpos, ypos, link_names[d])
plt.legend()
plt.show()

for d in range(len(link_names)):
    print('{}:\t{}'.format(link_names[d], np.round(xhist[-1,d], 4)))
    
