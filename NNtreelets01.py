# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:30:12 2017

@author: garrettsmith

Trying out ANN treelets.

Goals:
    1. Determiner treelet that settles on 'that' sg. or 'these' pl.
    2. Noun treelet that settles on 'dog' or 'dogs'

General equations:
    da/dt = -a + f(Wa + I)
"""

import numpy as np
import matplotlib.pyplot as plt


# Determiner treelet
# Dimensions: [+a, +some, +sg, +pl]
det_patterns = np.array([[1, -1, 1, -1], # a
                         [-1, 1, -1, 1]]).T # some
#det_patterns = (det_patterns + 1) / 2
W_det = (det_patterns @ det_patterns.T) / det_patterns.shape[1]
W_det += np.random.uniform(-0.01, 0.01, W_det.shape)
np.fill_diagonal(W_det, 0)
#det_init = np.array([1, 0, 0, 0]) # activating phonology for 'a'
det_init = np.array([-1, 1, 0, 0]) # activating phonology for 'some'

# Noun treelet
# Dimensions: [+dog, +cat, +a, +some, +sg, +pl]
noun_patterns = np.array([[1, -1, 1, -1, 1, -1], # dog
                          [1, -1, -1, 1, -1, 1], # dogs
                          [-1, 1, 1, -1, 1, -1], # cat
                          [-1, 1, -1, 1, -1, 1]]).T # cats
#noun_patterns = (noun_patterns + 1) / 2
W_noun = (noun_patterns @ noun_patterns.T) / noun_patterns.shape[1]
W_noun += np.random.uniform(-0.01, 0.01, W_noun.shape)
np.fill_diagonal(W_noun, 0)
#noun_init = np.zeros(noun_patterns.shape[0])
noun_init = np.random.uniform(-0.01, 0.01, noun_patterns[:,0].shape)

## Running both
tvec = np.arange(0.0, 500.0, 0.1)
det_hist = np.zeros((len(det_init), len(tvec)))
noun_hist = np.zeros((len(noun_init), len(tvec)))
det_hist[:,0] = det_init
noun_hist[:, 0] = noun_init

tstep = 0.01
det_overlap = np.zeros((len(tvec), det_patterns.shape[1]))
noun_overlap = np.zeros((len(tvec), noun_patterns.shape[1]))
link_strength = np.zeros(len(tvec))
for t in range(1, len(tvec)):
    link_strength[t] = det_hist[:, t-1] @ noun_hist[2:, t-1] / 4
    det_hist[:, t] = det_hist[:, t-1] + tstep * (-det_hist[:, t-1] + np.tanh(W_det @ det_hist[:, t-1] + link_strength[t] * noun_hist[2:,t-1]))
#    det_hist[:, t] = det_hist[:, t-1] + tstep * (-det_hist[:, t-1] + sig(W_det @ det_hist[:, t-1] + link_strength[t] * noun_hist[2:,t-1]))
    det_overlap[t,:] = (det_hist[:, t] @ det_patterns) / det_patterns.shape[0]
    
    input_from_det = np.zeros(noun_init.shape)
    input_from_det[2:] = det_hist[:, t-1]
    noun_hist[:, t] = noun_hist[:, t-1] + tstep * (-noun_hist[:, t-1] + np.tanh(W_noun @ noun_hist[:, t-1] + link_strength[t] * input_from_det))
#    noun_hist[:, t] = noun_hist[:, t-1] + tstep * (-noun_hist[:, t-1] + sig(W_noun @ noun_hist[:, t-1] + link_strength[t] * input_from_det))
    noun_overlap[t,:] = (noun_hist[:, t] @ noun_patterns) / noun_patterns.shape[0]
    
    if t == 1000:
        noun_hist[:,t] += np.array([1, 0, 0.5, 0.5, 0, 1]) # phonology for 'dogs'
#        noun_hist[:,t] += np.array([1, -1, 0, 0, 1, -1]) # phonology for 'dog'

# Plotting
det_labels = ['a', 'some']
for i in range(det_patterns.shape[1]):
    plt.plot(tvec, det_overlap[:, i], label = '{}'.format(det_labels[i]))
plt.legend()
plt.title('Determiner overlap')
plt.show()

noun_labels = ['dog', 'dogs', 'cat', 'cats']
for i in range(noun_patterns.shape[1]):
    plt.plot(tvec, noun_overlap[:, i], label = '{}'.format(noun_labels[i]))
plt.legend()
plt.title('Noun overlap')
plt.show()

plt.plot(link_strength)
plt.title('Link strength')
plt.show()