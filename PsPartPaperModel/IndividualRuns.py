# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:40 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize

nlinks = 6
k = 2
#k = 1.1
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']

# Params from CUNY 2017: made up
#box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
#group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
#lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
#many_N2 = np.array([0, 0, 0, 0.9, 0., 0.9])

# Feature matches
#box_of_N2 = np.array([3., 0, 3, 0, 3, 3])
#group_of_N2 = np.array([1., 1, 1, 1, 3, 3])
#lot_of_N2 = np.array([0., 3, 0, 3, 3, 3])
#many_N2 = np.array([0., 0, 0, 3, 0, 3])

# Manhattan distances
#box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
#group_of_N2 = np.array([2., 2, 2, 2, 0, 0])
#lot_of_N2 = np.array([3., 0, 3, 0, 0, 0])
#many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
group_of_N2 = np.array([2., 2, 2, 2, 2, 0])
lot_of_N2 = np.array([3., 0, 3, 0, 3, 0])
many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
# Similarity
#all_sents = normalize(np.exp(-np.array(all_sents)), norm='l2', axis=1)
all_sents = np.exp(-np.array(all_sents))

# Uncomment the one you want to try
#ipt = all_sents[0,] + np.random.uniform(0, 0.001, nlinks) # Container
ipt = all_sents[1,] + np.random.uniform(0, 0.001, nlinks) # Collection
#ipt = all_sents[2,] + np.random.uniform(0, 0.001, nlinks) # Measure
#all_sents[3,3] -= np.random.uniform(0, 0.001, 1)
#all_sents[3,5] -= np.random.uniform(0, 0.001, 1)
#ipt = all_sents[3,] # Quant

#tau = 1.
tau = 0.01
ntsteps = 10000
noisemag = 0.001
nreps = 100

#length = 0
length = 1
if length == 0:
    adj = 0.05
else:
    adj = 0.1
    
if ipt is many_N2:
    x0 = np.array([0, 0, 0, 0.101, 0., 0.001])
else:
    x0 = np.array([0.001]*nlinks)
    x0[0] += 0.1
    
xhist = np.zeros((ntsteps, nlinks))
xhist[0,] = x0
noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)

# Individual runs
#for t in range(1, ntsteps):
t = 0
while True:
    t += 1
    # Euler forward dynamics
#                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#                * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
#    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#    * (ipt - W @ (ipt * xhist[t-1,]))) + noise[t,:], -0.01, 1.01)
    xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
    * (1 - W @ xhist[t-1,])) + noise[t,:], -0.01, 1.01)

    if not np.all(ipt == all_sents[3,]):
        if t == 400:
            xhist[t,1] += adj
            xhist[t,2] += adj
        if t == 800:
            xhist[t,3:] += adj
    else:
        xhist[t,0:3] = np.clip(noise[t,0:3], -0.1, 1.1)
        xhist[t,4] = np.clip(noise[t,4], -0.01, 1.01)
        if t == 400:
            xhist[t,5] += adj
    if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
        break
    elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
        break
    elif t == ntsteps:
        break


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
#    xpos = d * (len(xhist[:,d]) / len(link_names))
#    ypos = xhist[xpos, d]
#    plt.text(xpos, ypos, link_names[d])
plt.legend()
plt.title('Individual link competition run')
plt.show()

