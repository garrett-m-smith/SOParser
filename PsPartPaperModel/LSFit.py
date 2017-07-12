# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:18:37 2017

@author: garrettsmith
"""

# Trying out simple least squares fitting of model parameters to the
# verb choice data

# Following https://python4mpia.github.io/fitting_data/least-squares-fitting.html
# Doesn't really work....

import numpy as np
import scipy.optimize as optim

# defining the dynamics
def dyn(noisemag, adj, k, x0, word_timing, tau):
    ntsteps = 5000
#    nreps = 1408 # total number of trials from human data
    nreps = 100
    nlinks = 6
#    x0 = np.array([0.001] * nlinks)
    x0 = np.array([x0] * nlinks)
#    x0[0] = 0.01
    x0[0] += adj
    noisemag = noisemag
#    tau = 0.01
    tau = tau
    box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
    group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
    lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
    all_sents = [box_of_N2, group_of_N2, lot_of_N2]
    data = np.zeros((2, len(all_sents), 3))
    k = k
    W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])
    for length in range(2):
        if length == 0:
#            print('Starting -PP')
            adj = adj / 2
        else:
#            print('Starting +PP')
            adj = adj
        
        for sent in range(len(all_sents)):
            # Set current input
            ipt = all_sents[sent]
#            x0[0] = ipt[0]
#            print('Starting N1 Type {}'.format(sent))
            
            for rep in range(nreps):
#                if rep % 50 == 0:
#                    print('Run {}'.format(rep))
                # For each repetition, reset history and noise
                xhist = np.zeros((ntsteps, nlinks))
                xhist[0,] = x0
                noise = np.random.normal(0, noisemag, xhist.shape)
                for t in range(1, ntsteps):
                    # Euler forward dynamics
                    xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
                    * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
    
                    if t == word_timing:
                        xhist[t,2] += adj
                        xhist[t,1] += adj
                    if t == 2*word_timing:
                        xhist[t,3:] += adj

            # Tallying the final states        
            final = np.round(xhist[-1,])        
            if np.all(final == [1, 0, 1, 0, 1, 0]):
                data[length, sent, 0] += 1
            elif np.all(final == [0, 1, 0, 1, 0, 1]):
                data[length, sent, 1] += 1
            else:
                data[length, sent, 2] += 1
    return data / np.sum(data)

#test = dyn(0.1, 0.01, 2)

human_data = np.zeros((2, 3, 3))
human_data[0,:,:] = np.array([[222, 14, 0],
                             [189, 46, 0],
                             [99, 134, 0]])
human_data[1,:,:] = np.array([[137, 97, 0],
                             [90, 145, 0],
                             [34, 201, 0]])
human_data = human_data / np.sum(human_data)

def obj_fn(params):
    model_data = dyn(*params)
    return np.sum((model_data - human_data)**2)

noise0 = 1.0
adj0 = 0.1
k0 = 2.
x0 = 0.01
word_timing0 = 100
tau0 = 0.1
params0 = [noise0, adj0, k0, x0, word_timing0, tau0]
fit = optim.fmin(obj_fn, params0)
#fit = optim.fmin(obj_fn, params0, maxiter=50)
# best yet after 50 iterations and noise0 = 0.1, adj0 = 0.01, k0 = 2., 
# x0 = 0.001, word_timing0 = 100, tau0 = 0.01:
# array([  9.95400568e-02,   9.99304307e-03,   2.01017523e+00,
#         1.01012903e-03,   9.92545668e+01])

