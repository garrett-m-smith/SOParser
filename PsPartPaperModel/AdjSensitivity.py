# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:54:12 2017

@author: garrettsmith
"""

# Just the dynamics

import numpy as np
import matplotlib.pyplot as plt

def dyn(k0, adj0):
    nlinks = 6
    n1types = 4
    npp = 2

    box_of_N2 = np.array([0., 1, 1, 1, 1, 0])
    group_of_N2 = np.array([2., 1, 1, 1, 1, 0])
    lot_of_N2 = np.array([3., 1, 1, 1, 1, 0])
    many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

    all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
    all_sents = np.exp(-np.array(all_sents))

    ## Monte Carlo
    tau = 0.01
    ntsteps = 10000
    noisemag = 0.001
    nreps = 50
    k = k0
    W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])
    
    data = np.zeros((npp, n1types, 3))

    for length in range(npp):
        if length == 0:
#            print('Starting -PP')
#            adj = 0.0
            adj = adj0/2.
        else:
#            print('Starting +PP')
            adj = adj0
        
        for sent in range(n1types):
            # Set current input
            ipt = all_sents[sent,]
#            print('\tStarting sentence {}'.format(sent))
    
            for rep in range(nreps):
                # For each repetition, reset history and noise
                if sent == 3:
                    # Minus to keep < 1
                    ipt = all_sents[3,]
                    ipt[3] -= np.random.uniform(0, 0.001, 1)
                    ipt[5] -= np.random.uniform(0, 0.001, 1)
                    x0 = np.array([0, 0, 0, 0.001, 0., 0.001])
                    x0[3] += adj0
                else:
                    ipt = all_sents[sent,] + np.random.uniform(0, 0.001, nlinks)
                    x0 = np.array([0.001]*nlinks)
                    x0[0] += adj0
                xhist = np.zeros((ntsteps, nlinks))
                xhist[0,] = x0
                noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)
            
                t = 0
                while True:
                    t += 1
                    xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
                    * (1 - W @ xhist[t-1,])) + noise[t,:], -0.01, 1.01)

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
                        xhist[t,0:3] = np.clip(noise[t,0:3], -0.1, 1.1)
                        xhist[t,4] = np.clip(noise[t,4], -0.01, 1.01)
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

def sse(model_data):
    return np.sum((model_data - human_data)**2)

def attr(model_data):
    return model_data[1, :, 1] - model_data[0, :, 1]

# Doing sensitivity
adj_vec = np.linspace(0.0, 1.0, 100)
errors = np.zeros(len(adj_vec))
attr_eff = np.zeros((len(adj_vec), 4))
for i in range(len(adj_vec)):
    if i % 10 == 0: print(i)
    distr = dyn(k0=2, adj0=adj_vec[i])
    errors[i] = sse(distr)
    attr_eff[i,] = attr(distr)

plt.figure(figsize=(6, 4))
plt.plot(adj_vec, errors)
plt.ylabel('SSE')
plt.xlabel('Adj')
plt.title('Effect of boost magnitude on error')
plt.show()

labels = ['Containers', 'Collections', 'Measures', 'Quantifiers']
plt.figure(figsize=(6, 4))
for i in range(4):
    plt.plot(adj_vec, attr_eff[:,i], label=labels[i])
plt.legend()
plt.ylabel('+PP - -PP')
plt.xlabel('Adj')
plt.title('Effect of boost magnitude on agreement attraction')
plt.show()
