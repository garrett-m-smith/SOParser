# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:40 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt

nlinks = 6
k = 2
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']

# Params from CUNY 2017
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])

# Uncomment the one you want to try
ipt = box_of_N2 # Container
#ipt = group_of_N2 # Collection
#ipt = lot_of_N2 # Measure Phrase

tau = 0.01
nsec = 50
tvec = np.linspace(0, nsec, nsec/tau + 1)
x0 = np.array([0.2] * nlinks)
adj = 2.

# Setting first word to between its current state and 1
x0[0] = x0[0] + (1 - x0[0]) / adj
  
# Creating history fector and initializing noise
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0
noisemag = 1.5
noise = np.random.normal(0, noisemag, xhist.shape)


# Individual runs
for t in range(1, len(tvec)):
    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
    * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
    
    # Introducing the preposition
    if t == 100:
        # Turn boost P-P(N1) = intro Prep
        xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
        # Also turn on its competitor: N1-N(P)
        xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
    if t == 200:
        # Intro N2-related links: P-Det(N2)
        xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
        # N2-N(P)
        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
        # N2-N(V)
        xhist[t,4] = xhist[t,5] + (1 - xhist[t,5]) / adj

# If you want to save individual trajectories as CSVs
#np.savetxt('box-N1-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('group-N1-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('group-N2-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('lot-N2-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')

# Printing out the final vlaues of each link
for d in range(len(link_names)):
    print('{}:\t{}'.format(link_names[d], np.round(xhist[-1,d], 4)))

plt.figure()
plt.ylim(-0.1, 1.1)
for d in range(xhist.shape[1]):
    plt.plot(xhist[:,d], label = link_names[d])
    xpos = d * (len(xhist[:,d]) / len(link_names))
    ypos = xhist[xpos, d]
    plt.text(xpos, ypos, link_names[d])
plt.legend()
plt.title('Individual link competition run')
plt.show()

