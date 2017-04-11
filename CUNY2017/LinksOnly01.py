# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith

Introducing feature vectors
"""

import numpy as np
#import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

nlinks = 6
link_names = ['N1->N(V)', 'N1->N(P)', 'P->P(N1)', 'P->Det(N2)', 
'N2->N(P)', 'N2->N(V)']


# Setting the LV growth rates to plausible values given our feature cline
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])

# Interaction matrix: specifies which links enter into WTA competitions. The
# parameter k determines the relative strength of inhibition from other links
# to a link's self-inhibition
k = 2.
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, 2, 1]])


## Individual runs
#ipt = box_of_N2
#ipt = group_of_N2
#ipt = lot_of_N2

## Time constant of the dynamics
#tau = 0.1
#
## Length of simulation in arbitrary time units
#nsec = 50
#
## Time vector for simulation
#tvec = np.linspace(0, nsec, nsec/tau + 1)
#
## Initial conditions: here all at 0.2
#x0 = np.array([0.2] * nlinks)
#
## How much to bump up the activation when a phonological form comes in
#adj = 2.
#
## Setting first word to between its current state and 1
#x0[0] = x0[0] + (1 - x0[0]) / adj
#xhist = np.zeros((len(tvec), nlinks))
#xhist[0,] = x0
#noisemag = 0.1
#noise = np.random.normal(0, noisemag, xhist.shape)

# Individual runs
#for t in range(1, len(tvec)):
#    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#    * (ipt - W @ (ipt * xhist[t-1,]) + noise[t,:])), -0.1, 1.1)
##    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
##    * (ipt - W @ (ipt * xhist[t-1,]))), -0.1, 1.1)
#    
#    if t == 100:
#        # Turn boost P-P(N1) = intro Prep
#        xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
##        xhist[t,2] = 0.7
#        # Also turn on its competitor: N1-N(P)
#        xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
##        xhist[t,1] = 0.7
#    if t == 200:
#        # Intro N2-related links: P-Det(N2)
#        xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
##        xhist[t,3] = 0.7
#        # N2-N(P)
#        xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
##        xhist[t,4] = 0.7
#        # N2-N(V)
#        xhist[t,5] = xhist[t,5] + (1 - xhist[t,5]) / adj
##        xhist[t,5] = 0.7
#
#
#plt.figure()
#plt.ylim(-0.1, 1.1)
#for d in range(xhist.shape[1]):
#    plt.plot(xhist[:,d], label = link_names[d])
#    xpos = d * (len(xhist[:,d]) / len(link_names))
#    ypos = xhist[xpos, d]
#    plt.text(xpos, ypos, link_names[d])
##plt.legend()
#plt.legend(bbox_to_anchor = (1, 1.03))
#plt.show()
#
#for d in range(len(link_names)):
#    print('{}:\t{}'.format(link_names[d], np.round(xhist[-1,d], 1)))
    

## Monte Carlo
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

nreps = 1000
all_sents = [box_of_N2, group_of_N2, lot_of_N2]

# For saving final states
data = np.zeros((len(all_sents), 3))

for sent in range(len(all_sents)):
    # Set current input
    ipt = all_sents[sent]
    x0[0] = ipt[0]
    print('Starting sentence {}'.format(sent))
    
    for rep in range(nreps):
        # For each repetition, reset history and noise
        xhist = np.zeros((len(tvec), nlinks))
        xhist[0,] = x0
        noise = np.random.normal(0, noisemag, xhist.shape)
        
        for t in range(1, len(tvec)):
            # Euler forward dynamics
            xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
            * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
    
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
                xhist[t,5] = xhist[t,5] + (1 - xhist[t,5]) / adj

        # Tallying the final states        
        final = np.round(xhist[-1,])        
        if np.all(final == [1, 0, 1, 0, 1, 0]):
            data[sent,0] += 1
        elif np.all(final == [0., 1, 0, 1, 0, 1]):
            data[sent,1] += 1
        else:
            data[sent,2] += 1

print(data)

# Reported in CUNY 2017 talk
#                   N1     N2    Other
# Containers    [[ 982.   10.    8.]
# Collections   [ 214.  761.   25.]
# Measures      [   0.  997.    3.]]
