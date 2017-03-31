# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith

Introducing feature vectors
"""

import numpy as np
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

nlinks = 6
link_names = ['N1->N(V)', 'N1->N(P)', 'P->P(N1)', 'P->Det(N2)', 
'N2->N(P)', 'N2->N(V)']

## Possibly add number features?
## Features: +-Container, +-PhysConfig +-ConcreteN2, +-N, +-OtherPOS
## Head attachment sites
#nfeat = 5
## Noun attch. site of verbs like 'box'
#N_V = np.array([1., 1, 1, 1, 0])
## Noun attch. site of Prepositions like N1s without Concrete N2
#N1_P = np.array([1., 1, 1, 1, 0])
#N2_P = np.array([0, 0, 0, 1, 0])
## Preposition attch. site of N1 likes things that aren't nouns
#P_N1 = np.array([0., 0, 0, 0, 1])
## Det attch. site of N2 likes things that aren't nouns
#Det_N2 = np.array([0., 0, 0, 0, 1])
#
## Sending attachment sites
#box = np.array([1, 1, 1, 1, 0])
#group = np.array([0, 1, 1, 1, 0])
#lot = np.array([0, 0, 0, 1, 0])
#N2 = np.array([1, 1, 0, 1, 0])
#of = np.array([0., 0, 0, 0, 1])
#
#box_N_V = box @ N_V / nfeat
#box_N_P = box @ N1_P / nfeat
#group_N_V = group @ N_V / nfeat
#group_N_P = group @ N1_P / nfeat
#lot_N_V = lot @ N_V / nfeat
#lot_N_P = lot @ N1_P / nfeat
#of_P_N1 = of @ P_N1 / nfeat
#of_Det_N2 = of @ Det_N2 / nfeat
#N2_N_P = N2 @ N2_P / nfeat
#N2_N_V = N2 @ N_V / nfeat

# box of N2s
#box_of_N2 = np.array([box_N_V, box_N_P, of_P_N1, of_Det_N2, N2_N_P, N2_N_V])
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
#group_of_N2 = np.array([group_N_V, group_N_P, of_P_N1, of_Det_N2, N2_N_P, N2_N_V])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
#lot_of_N2 = np.array([lot_N_V, lot_N_P, of_P_N1, of_Det_N2, N2_N_P, N2_N_V])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])

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
k = 2.
# Limited interactions
#W = np.array([[1., k, 0, 0, 0, k],
#              [k, 1, k, 0, k, 0],
#              [0, k, 1, k, 0, 0],
#              [0, 0, k, 1, k, 0],
#              [0, k, 0, k, 1, k],
#              [k, 0, 0, 0, k, 1]])

# More constraining interactions
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, 2, 1]])


tau = 0.1
nsec = 50
tvec = np.linspace(0, nsec, nsec/tau + 1)
x0 = np.array([0.2] * nlinks)
adj = 2.
# Setting first word to between its current state and 1
x0[0] = x0[0] + (1 - x0[0]) / adj
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0
noisemag = 0.1
noise = np.random.normal(0, noisemag, xhist.shape)

#ipt = box_of_N2
#ipt = group_of_N2
ipt = lot_of_N2

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
    

# Monte Carlo
tau = 0.01
nsec = 50
tvec = np.linspace(0, nsec, nsec/tau + 1)
x0 = np.array([0.2] * nlinks)
adj = 2.
# Setting first word to between its current state and 1
x0[0] = x0[0] + (1 - x0[0]) / adj
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0
#noisemag = 1. # good!
#noisemag = 2. # better balance, but with more Others
noisemag = 1.5
noise = np.random.normal(0, noisemag, xhist.shape)

nreps = 1000
all_sents = [box_of_N2, group_of_N2, lot_of_N2]
data = np.zeros((len(all_sents), 3))
other_parses = np.zeros(nlinks)
for sent in range(len(all_sents)):
    ipt = all_sents[sent]
    x0[0] = ipt[0]
    print('Starting sentence {}'.format(sent))
    for rep in range(nreps):
        xhist = np.zeros((len(tvec), nlinks))
        xhist[0,] = x0
#        xhist[0,] = np.random.uniform(0.1, 0.2, xhist.shape[1])
        noise = np.random.normal(0, noisemag, xhist.shape)
        
        for t in range(1, len(tvec)):
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
        
        final = np.round(xhist[-1,])
        
#        if np.allclose(final, np.array([1., 0, 1, 0, 1, 0]), rtol = 0.4):
        if np.all(final == [1, 0, 1, 0, 1, 0]):
            data[sent,0] += 1
        elif np.all(final == [0., 1, 0, 1, 0, 1]):
#        elif np.allclose(final, np.array([0., 1, 0, 1, 0, 1]), rtol = 0.4):
            data[sent,1] += 1
        else:
            data[sent,2] += 1
#            other_parses = np.vstack((other_parses, final))

print(data)

# Reported in CUNY 2017 talk
#[[ 982.   10.    8.]
# [ 214.  761.   25.]
# [   0.  997.    3.]]
# Good run with 5000 sims
# tau = 0.01, x0 = 0.2, adj = 2, noisemag = 1., additive noise
#[[  4.99700000e+03   3.00000000e+00   0.00000000e+00]
# [  8.66000000e+02   4.13200000e+03   2.00000000e+00]
# [  0.00000000e+00   5.00000000e+03   0.00000000e+00]]