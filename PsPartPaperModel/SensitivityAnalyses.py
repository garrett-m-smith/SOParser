# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:54:12 2017

@author: garrettsmith
"""

# Just the dynamics

import numpy as np
import matplotlib.pyplot as plt

def dyn(k0, adj0, noise0):
    nlinks = 6

    box_of_N2_pp = np.array([0., 1, 1, 1, 1, 0])
    group_of_N2_pp = np.array([1., 1, 1, 1, 1, 0])
    lot_of_N2_pp = np.array([3., 1, 1, 1, 1, 0])
    many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

    box_of_N2_no = np.array([0., 2, 2, 2, 2, 1])
    group_of_N2_no = np.array([1., 2, 2, 2, 2, 1])
    lot_of_N2_no = np.array([3., 2, 2, 2, 2, 1])
    many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])
    all_sents = [box_of_N2_pp, group_of_N2_pp, lot_of_N2_pp, many_N2_pp, box_of_N2_no, group_of_N2_no, lot_of_N2_no, many_N2_no]
    all_sents = np.exp(-np.array(all_sents))

    ## Monte Carlo
    tau = 0.01
    ntsteps = 10000
    noisemag = noise0
    nreps = 50
    k = k0
    W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])
    
    data = np.zeros((len(all_sents), 3))
    for sent in range(all_sents.shape[0]):
        ipt = all_sents[sent,]
#        print('\tStarting sentence {}'.format(sent))
    
        for rep in range(nreps):
            # For each repetition, reset history and noise
            if sent == 3 or sent == 7:
                x0 = np.array([0, 0, 0, 0.101, 0., 0.001])
            else:
                x0 = np.array([0.001]*nlinks)
                x0[0] += 0.1
#               x0[0] += 0.01
            xhist = np.zeros((ntsteps, nlinks))
            xhist[0,] = x0
            noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)
            
            t = 0
            while t < ntsteps-1:
                t += 1
                xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
                * (1 - W @ xhist[t-1,])) + noise[t-1,:], -0.01, 1.01)

                if sent < 3:
                    if t == 400:
                        xhist[t,1] += adj0
                        xhist[t,2] += adj0
                    if t == 800:
                        xhist[t,3:] += adj0
                    if t >= 1200:
                        if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                            data[sent, 0] += 1
                            break
                        elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                            data[sent, 1] += 1
                            break
                        elif (t+1) == ntsteps:
                            data[sent, 2] += 1
                            break
                elif sent == 3:
                    xhist[t, 0:3] = 0
                    xhist[t, 4] = 0
                    if t == 400:
                        xhist[t,5] += adj0
                    if t >= 800:
                        if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                            data[sent, 0] += 1
                            break
                        elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                            data[sent, 1] += 1
                            break
                        elif (t+1) == ntsteps:
                            data[sent, 2] += 1
                            break
                elif sent > 3 and sent < 7:
                # Assuming the elided material all comes in at once
                    if t > 400:
                        if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                            data[sent, 0] += 1
                            break
                        elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                            data[sent, 1] += 1
                            break
                        elif (t+1) == ntsteps:
                            data[sent, 2] += 1
                            break
                else:
                    xhist[t, 0:3] = 0
                    xhist[t, 4] = 0            
                    if t > 400:
                        if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
                            data[sent, 0] += 1
                            break
                        elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
                            data[sent, 1] += 1
                            break
                        elif (t+1) == ntsteps:
                            data[sent, 2] += 1
                            break
    return data / nreps

human_data = np.array([[137, 97, 0],
                       [90, 145, 0],
                       [34, 201, 0],
                       [9, 227, 0],
                       [222, 14, 0],
                       [189, 46, 0],
                       [99, 134, 0],
                       [27, 206, 0]])
human_data = human_data / human_data.sum(axis=1)[:,None]
#for i in range(2):
#    for j in range(4):
#        human_data[i,j,:] = human_data[i,j,:] / human_data[i,j,:].sum()

def sse(model_data):
    return np.sum((model_data - human_data)**2)

#def attr(model_data):
#    return model_data[1, :, 1] - model_data[0, :, 1]

def agr_attr(model_data):
    """Returns the average agr. attr. effect for Cont. through Meas."""
    return np.sum(model_data[0:3,1] - model_data[4:7,1]) / 3.

# Doing sensitivity
#noise_vec = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
noise_vec = np.linspace(0.0001, 0.1, 40)
k_vec = np.linspace(0.35, 2., 40)
#adj_vec = np.linspace(0.0, 1.0, 100)
adj = 0.1 # 
#errors = np.zeros(len(adj_vec))
errors = np.zeros((len(noise_vec), len(k_vec)))
#attr_eff = np.zeros((len(adj_vec), 4))
attr_eff = np.zeros((len(noise_vec), len(k_vec)))
#for i in range(len(adj_vec)):
for i in range(len(noise_vec)):
    for j in range(len(k_vec)):
        if j % 10 == 0: print('Noise: {}\tk: {}'.format(noise_vec[i], k_vec[j]))
        distr = dyn(k0=k_vec[j], adj0=adj, noise0=noise_vec[i])
        errors[i,j] = sse(distr)
        attr_eff[i,j] = agr_attr(distr)

plt.figure(figsize=(6, 6))
plt.pcolor(noise_vec, k_vec, errors.T)
plt.colorbar()
plt.title('SSE')
plt.ylabel('Competition parameter k')
plt.xlabel('Noise magnitude')
plt.show()

plt.figure(figsize=(6, 6))
plt.pcolor(noise_vec, k_vec, attr_eff.T)
plt.colorbar()
plt.title('Avg. agreement attraction effect\nContainers through Measures')
plt.ylabel('Competition parameter k')
plt.xlabel('Noise magnitude')
plt.show()

attr_eff2 = attr_eff
attr_eff2[attr_eff2 < 0] = 0
plt.figure(figsize=(6, 6))
plt.pcolor(noise_vec, k_vec, attr_eff2.T)
plt.colorbar()
plt.title('Avg. agreement attraction effect\nContainers through Measures')
plt.ylabel('Competition parameter k')
plt.xlabel('Noise magnitude')
plt.show()

#plt.figure(figsize=(6, 4))
#plt.plot(adj_vec, errors)
#plt.ylabel('SSE')
#plt.xlabel('Adj')
#plt.title('Effect of boost magnitude on error')
#plt.show()
#
#labels = ['Containers', 'Collections', 'Measures', 'Quantifiers']
#plt.figure(figsize=(6, 4))
#for i in range(4):
#    plt.plot(adj_vec, attr_eff[:,i], label=labels[i])
#plt.legend()
#plt.ylabel('+PP - -PP')
#plt.xlabel('Adj')
#plt.title('Effect of boost magnitude on agreement attraction')
#plt.show()
