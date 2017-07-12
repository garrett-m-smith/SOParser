# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith
"""

import numpy as np

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']


# Setting the LV growth rates to plausible values given our feature cline.
# Each dimension corresponds to the links in link_labels above.
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
#box_of_N2 = np.array([1., 1/3., 1, 1, 1, 1])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
#group_of_N2 = np.array([0.5, 1/3., 0.5, 1, 1, 1])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
#lot_of_N2 = np.array([1/3., 1, 1/3., 1, 1, 1])

# Interaction matrix: specifies which links enter into WTA competitions. The
# parameter k determines the relative strength of inhibition from other links
# to a link's self-inhibition
#k = 2.
k = 1.1
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

## Monte Carlo
tau = 0.01
#nsec = 50
#tvec = np.linspace(0, nsec, int(nsec/tau + 1))
ntsteps = 10000
#x0 = np.array([0.2] * nlinks)
x0 = np.array([0.001] * nlinks)
adj = 2.

# Setting first word to between its current state and 1
#x0[0] = x0[0] + (1 - x0[0]) / adj
#x0[0] = 0.01
x0[0] = 0.1
  
# Creating history fector and initializing noise
#xhist = np.zeros((len(tvec), nlinks))
xhist = np.zeros((ntsteps, nlinks))
xhist[0,] = x0
#noisemag = 1.
noisemag = 0.5
noise = np.random.normal(0, noisemag, xhist.shape)

nreps = 1000
all_sents = [box_of_N2, group_of_N2, lot_of_N2]

# For saving final states; dims: length, N1 Type, parse type(N1, N2, other)
data = np.zeros((2, len(all_sents), 3))

for length in range(2):
    if length == 0:
        print('Starting -PP')
        # Half the boost if short
#        adj = 4.
        adj = 0.05
    else:
        print('Starting +PP')
#        adj = 2.
        adj = 0.1
        
    for sent in range(len(all_sents)):
    # Set current input
        ipt = all_sents[sent]
#        x0[0] = ipt[0]
        print('Starting sentence {}'.format(sent))
    
        for rep in range(nreps):
        # For each repetition, reset history and noise
            xhist = np.zeros((ntsteps, nlinks))
            xhist[0,] = x0
            noise = np.random.normal(0, noisemag, xhist.shape)
        
            for t in range(1, ntsteps):
                # Euler forward dynamics
                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
                * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
#                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#                * (ipt - W @ (ipt * xhist[t-1,]) + noise[t,:])), -0.1, 1.1)
    
                if t == 25:
#                if t == 300:
#                    xhist[t,1:4] += adj
                    xhist[t,2] += adj
                    xhist[t,1] += adj
                    # Turn boost P-P(N1) = intro Prep
#                    xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
                    # Also turn on its competitor: N1-N(P)
#                    xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
                if t == 50:
#                if t == 600:
                    xhist[t,3:] += adj
                    # Intro N2-related links: P-Det(N2)
#                    xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
                    # N2-N(P)
#                    xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
                    # N2-N(V)
#                    xhist[t,5] = xhist[t,5] + (1 - xhist[t,5]) / adj

            # Tallying the final states        
            final = np.round(xhist[-1,])        
            if np.all(final == [1, 0, 1, 0, 1, 0]):
                data[length, sent, 0] += 1
            elif np.all(final == [0, 1, 0, 1, 0, 1]):
                data[length, sent, 1] += 1
            else:
                data[length, sent, 2] += 1

#print(data)
print('-PP:')
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}'.format(*data[0]))
print('+PP:')
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}'.format(*data[1]))
    

# Best yet:
# -PP:
#Containers:     [ 817.  131.   52.]
#Collections:    [ 455.  518.   27.]
#Measures:       [  80.  835.   85.]
#+PP:
#Containers:     [ 891.  100.    9.]
#Collections:    [ 251.  741.    8.]
#Measures:       [   7.  970.   23.]