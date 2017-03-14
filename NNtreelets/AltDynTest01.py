# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:33:32 2017

@author: garrettsmith

Trying out 2LC dyn:
    dx/dt = nx(1-x)
    n = net input
"""

import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    return 1 / (1 + np.exp(-x))

det_patterns = np.array([[1, 0, 0, 0, 1, 0], # a
                         [0, 1, 0, 0, 0, 1]]).T # these

# Setting weights by hand:
W_det = np.array([[1, -1, 0, 0, 1, -1],
                  [-1, 1, 0, 0, 0, 0],
                  [0, 0, 1, -1, 0, 0],
                  [0, 0, -1, 1, 0, 0],
                  [1, 0, 0, 0, 1, -1],
                  [-1, 0, 0, 0, -1, 1]])
#W_det = (W_det + 1) / 2
W_det = (det_patterns @ det_patterns.T) / det_patterns.shape[1]
#W_det = 0.05 * W_det
# Hebbian/covariance matrix for weights
#W_det = (det_patterns @ det_patterns.T) / det_patterns.shape[1]
# Adding noise should eliminate spurious attractors: HKP 91,
# Crisanti & Sompolinsky 1987
#W_det += np.random.uniform(-0.01, 0.01, W_det.shape)
#np.fill_diagonal(W_det, 0)

#det_init = np.array([0, 1, 0, 0, 0, 1]) # activating phonology for 'these'
det_init = np.random.uniform(0, 0.1, det_patterns.shape[0])

# Trying it out:
tstep = 0.001
tvec = np.arange(0.0, 30.0, tstep)
det_hist = np.zeros((len(tvec), len(det_init)))
det_hist[0,] = det_init
det_sim = np.zeros((len(tvec), det_patterns.shape[1]))

for t in range(1, len(tvec)):
#    det_hist[t,] = det_hist[t-1,] + tstep * ((W_det @ det_hist[t-1,]) * det_hist[t-1,]
#        * (1 - det_hist[t-1,]))
    det_hist[t,] = det_hist[t-1,] + tstep * (-det_hist[t-1,] + sig(W_det @ det_hist[t-1,]))
    det_sim[t,] = np.exp(-np.linalg.norm(det_patterns.T - det_hist[t-1,], axis = 1)**2)
    
plt.plot(tvec, det_sim)