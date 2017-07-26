# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:54:25 2017

@author: garrettsmith
"""

# Alternative dynamics
import numpy as np
import matplotlib.pyplot as plt

nlinks = 6
k = 2.
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
box_of_N2 = np.array([0., 1, 1, 1, 1, 0])
group_of_N2 = np.array([2., 1, 1, 1, 1, 0])
lot_of_N2 = np.array([3., 1, 1, 1, 1, 0])
many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
all_sents = np.exp(-np.array(all_sents))

ipt = all_sents[1,]

tau = 0.01
ntsteps = 20000
noisemag = 0.001
nreps = 100

# The change
r0 = np.array([-0.01]*nlinks)
r0[0] = 1.
length = 0
#length = 1
if length == 0:
    r_post = 1./k + 0.01 # from Till's paper
#    r_post = 0.1
else:
    r_post = 1.

xhist = np.zeros((ntsteps, nlinks))
xhist[0,] = np.array([0.01]*nlinks)
noise = np.sqrt(tau * noisemag) * np.random.normal(0, 1, (ntsteps, nlinks))
for t in range(1,ntsteps):
    if t == 400:
        r0[1:3] = r_post
        xhist[t-1,1:3] += 0.1
    elif t == 800:
        r0[3:] = r_post
        xhist[t-1,3:] += 0.1
    xhist[t,] = (xhist[t-1,] + tau * (ipt * xhist[t-1,] * (r0 - W @ xhist[t-1,]))
        + xhist[t-1,]*noise[t,])
#    xhist[t,] = np.clip(xhist[t,], -0.01, 1.1)

plt.figure(figsize=(10, 8))
for i in range(nlinks):
    plt.plot(xhist[:,i], label=link_names[i])
plt.legend()
plt.ylim(-0.02, 1.15)
plt.show()
