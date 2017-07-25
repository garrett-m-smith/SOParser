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
from scipy.optimize import minimize
#import profile
#from numba import jit
#import matplotlib.pyplot as plt

# Basics
nlinks = 6
pp = ['-PP', '+PP']
#box_of_N2 = np.array([0., 1, 1, 1, 1, 0])
#group_of_N2 = np.array([2., 1, 1, 1, 1, 0])
#lot_of_N2 = np.array([3., 1, 1, 1, 1, 0])
#many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])
box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
group_of_N2 = np.array([2., 2, 2, 2, 2, 0])
lot_of_N2 = np.array([3., 0, 3, 0, 3, 0])
many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])
all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
all_sents = np.exp(-np.array(all_sents))
ntsteps = 10000
#nreps = 250 # to approx. the number in each row of human data
nreps = 100
#nreps = 1
tau = 0.01
noisemag = 0.001
clip = 0.01

# defining the dynamics
#def dyn(noisemag0, adj0, k0):
#@jit
def dyn(adj0, k0):
    k = k0
    W = np.array([[1, k, 0, k, 0, k],
                  [k, 1, k, 0, k, 0],
                  [0, k, 1, k, 0, k],
                  [k, 0, k, 1, k, 0],
                  [0, k, 0, k, 1, k],
                  [k, 0, k, 0, k, 1]])
#    noisemag = noisemag0
    data = np.zeros((len(pp), len(all_sents), 3))
    # Pre-generating feature & link noise
    feat_noise = np.random.uniform(0, 0.001,
                                   (len(pp), all_sents.shape[0], nreps, nlinks))
    link_noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1,
                                  (len(pp), all_sents.shape[0],
                                   nreps, ntsteps, nlinks))
    print('Starting run...')
    for length in range(len(pp)):
        if length == 0:
            adj = adj0 / 2
        else:
            adj = adj0
        
        for sent in range(all_sents.shape[0]):
            # Set current input
            ipt = all_sents[sent,]
    
            for rep in range(nreps):
                # For each repetition, reset history and noise
                if sent == 3:
                    ipt = all_sents[3,]
                    # Minus to keep < 1
#                    all_sents[3,3] -= np.random.uniform(0, 0.001, 1)
#                    all_sents[3,5] -= np.random.uniform(0, 0.001, 1)
                    ipt[3] -= feat_noise[length, sent, rep, 3]
                    ipt[5] -= feat_noise[length, sent, rep, 5]
                    x0 = np.array([0, 0, 0, 0.001, 0., 0.001])
                    x0[3] += adj0
                    x0[-1] += adj0
                else:
#                    ipt = all_sents[senta,] + np.random.uniform(0, 0.001, nlinks)
                    ipt = all_sents[sent,] + feat_noise[length, sent, rep,:]
                    ipt = np.clip(ipt, 0., 1., out=ipt)
                    x0 = np.array([0.001]*nlinks)
                    x0[0] += adj0
            
                xhist = np.zeros((ntsteps, nlinks))
                xhist[0,] = x0
#                noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)
            
                t = 0
                while True:
                    t += 1
                    xprev = xhist[t-1,]
                    curr_noise = link_noise[length, sent, rep, t,:]
#                    xhist[t,:] = (xhist[t-1,] + tau * (ipt * xhist[t-1,] 
#                    * (1 - W @ xhist[t-1,]))
#                    + link_noise[length, sent, rep, t,:])
                    xnext = (xprev + tau * (ipt * xprev * (1 - W @ xprev))
                    + curr_noise)
                    xhist[t,:] = np.clip(xnext, 0-clip, 1+clip, out=xhist[t,:])

                    if sent != 3:
                        if t == 400:
                            xhist[t,1] += adj
                            xhist[t,2] += adj
                        if t == 800:
                            xhist[t,3:] += adj
                        if t >= 1200:
                            if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                                data[length, sent, 0] += 1
                                break
                            elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                                data[length, sent, 1] += 1
                                break
                            elif (t+1) == ntsteps:
                                data[length, sent, 2] += 1
                                break
                    else:
                        xhist[t,0:3] = np.clip(link_noise[length, sent, rep, t, 0:3], 
                             0-clip, 1+clip)
                        xhist[t,4] = np.clip(link_noise[length, sent, rep, t,4],
                             0-clip, 1+clip)
                        if t == 400:
                            xhist[t,5] += adj
                        if t >= 800:
                            if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                                data[length, sent, 0] += 1
                                break
                            elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                                data[length, sent, 1] += 1
                                break
                            elif (t+1) == ntsteps:
                                data[length, sent, 2] += 1
                                break
    
    return data / nreps

#test = dyn(0.001, 0.1, 2.)
#profile.run('dyn(0.1, 2)', sort='tottime')

human_data = np.zeros((2, 4, 3))
human_data[0,:,:] = np.array([[222, 14, 0],
                             [189, 46, 0],
                             [99, 134, 0],
                             [27, 206, 0]])
human_data[1,:,:] = np.array([[137, 97, 0],
                             [90, 145, 0],
                             [34, 201, 0],
                             [9, 227, 0]])
for i in range(2):
    for j in range(4):
        human_data[i,j,:] = human_data[i,j,:] / human_data[i,j,:].sum()
        
#for i in range(2):
#    plt.plot(human_data[i,:, 1], 'o', label=pp[i])
#    plt.plot(test[i,:,1], '*', label=pp[i])

def obj_fn(params):
    model_data = dyn(*params)
    return np.sum((model_data - human_data)**2)

#noisemag0 = 0.001 # including seems to make things not change...
adj0 = 0.01
k0 = 2.
#params0 = [noisemag0, adj0, k0]
params0 = [adj0, k0]
fit = minimize(obj_fn, params0, method='Nelder-Mead', 
               options={'disp': True})#, 'maxiter': 20})
# after 20 iterations adj0 = 0.1, k0 = 2.: adj = 0.1, k = 2.1

#test = dyn(*fit[0])
