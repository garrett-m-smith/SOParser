# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:39:18 2017

@author: garrettsmith
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:33:48 2017

@author: garrettsmith

PsPart using Li et al. (2015) polynomial energy fn.a
"""

import numpy as np
# import matplotlib.pyplot as plt
from itertools import product


# Defining functions
def l2norm(x):
    return np.sqrt(x @ x)


def rbf(x, c, gamma):
    phi = np.exp(-l2norm(x-c)**2/gamma)
    return phi


#def dyn(x, c, gamma, weights, ipt):
def dyn(x, c, ipt):
    """Takes the current position, a list of attractors, and the gamma
    parameters as arguments and updates the state.
    """
    dx = np.zeros(len(x))
    to_sum = np.arange(0, len(c))
    for i in to_sum:
        prod = np.ones(len(x))
        to_prod = to_sum[to_sum != i]
        for j in to_prod:
            prod *= l2norm(x - c[j])**2
        dx += prod * (x - c[i])
    return 2*dx


#def energy(x, c, gamma, weights):
#    e = 0
#    for i in range(len(c)):
#        e += weights[i]*rbf(x, c[i], gamma)
#    return e


def speed(x, c, gamma, weights):
    ipt = [0.0]*len(x)
    #vel = dyn(x, c, gamma, weights, ipt)
    vel = dyn(x, c, ipt)
    return l2norm(vel)


ndim = 6
link_labels = ['N1-V', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']

box_of_N2_pp = np.array([0., 0, 1, 1, 0, 0])
group_of_N2_pp = np.array([1., 0, 1, 1, 0, 0])
lot_of_N2_pp = np.array([3., 0, 1, 1, 0, 0])
# many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

box_of_N2_no = np.array([0., 1, 2, 2, 1, 1])
group_of_N2_no = np.array([1., 1, 2, 2, 1, 1])
lot_of_N2_no = np.array([3., 1, 2, 2, 1, 1])
# many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])

all_sents = [box_of_N2_pp, group_of_N2_pp, lot_of_N2_pp, #many_N2_pp,
             box_of_N2_no, group_of_N2_no, lot_of_N2_no]#, many_N2_no]
#feat_match = np.exp(-np.array(all_sents))# + 2
feat_match = 0.3*np.exp(-np.array(all_sents))

orig = np.array([0.0]*ndim)
p1 = np.array([1.0, 0.0, 0, 0, 0, 0])
p2 = np.array([0.0, 1.0, 0, 0, 0, 0])
n1headed = np.array([1., 0, 1, 0, 1, 0])
n2headed = np.array([0., 1, 0, 1, 0, 1])
centers = [orig, p1, p2, n1headed, n2headed]

tau = 0.01
ntsteps = 1000
isi = 100
gamma = 0.33
noisemag = 0.15
nreps = 10
data = np.zeros((len(all_sents), 5))  # 5 centers

for sent in range(feat_match.shape[0]):
    print('\tStarting sentence {}'.format(sent))

    for rep in range(nreps):
        # For each repetition, reset history and noise
        should_break = False
        x0 = np.random.normal(0.25, 0.25*noisemag, ndim)
        xhist = np.zeros((ntsteps, ndim))
        xhist[0, ] = x0
        noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)

        t = 0
        #ipt = [0.0]*ndim
        ipt = feat_match[sent,].copy()
        ipt[2:, ] = 0.0
        while t < ntsteps-1:
            t += 1
#            xhist[t, ] = (xhist[t-1, ] + tau
#                          * dyn(xhist[t-1, ], centers, gamma,
#                                feat_match[sent], ipt)
#                          + noise[t-1, ])
            xhist[t, ] = (xhist[t-1, ] + tau
                          * dyn(xhist[t-1, ], centers, ipt)
                          + noise[t-1, ])
            if t == isi:
                #ipt = [0.5, 0.5, 0, 0, 0, 0]# - xhist[t-1, ]
                # SWITCH TO JUST BUMPING IT INSTANTANEOUSLY LIKE ORIG MODEL!
                ipt = feat_match[sent, ].copy()
                ipt[4:, ] = 0.0
            elif t == 2*isi:
                #ipt = [0.5]*ndim# - xhist[t-1, ]
                ipt = feat_match[sent, ].copy()
            elif t >= 3*isi and speed(xhist[t-1, ], centers,
                                      gamma, feat_match) < 1.:
                # Test to see which parse you're in
                ipt = [0.0]*ndim
                last = np.round(xhist[t-1, ].copy())
                for i in range(len(centers)):
                    if (last == centers[i]).all():
                        data[sent, i] += 1
                        should_break = True
                        break
                if should_break:
                    break

print(data)
