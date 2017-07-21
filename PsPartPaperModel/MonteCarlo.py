# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
pp = ['-PP', '+PP']

# Setting the LV growth rates to plausible values given our feature cline.
# Each dimension corresponds to the links in link_labels above.
#box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
#group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
#lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
#many_N2 = np.array([0, 0, 0, 0.9, 0., 0.9])

#box_of_N2 = np.array([3., 0, 3, 0, 3, 3])
#group_of_N2 = np.array([1., 1, 1, 1, 3, 3])
#lot_of_N2 = np.array([0., 3, 0, 3, 3, 3])
#many_N2 = np.array([0., 0, 0, 3, 0, 3])
#
#all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
#all_sents = minmax_scale(all_sents, feature_range=(0.1, 1.))

# Manhattan distances
box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
group_of_N2 = np.array([2., 2, 2, 2, 0, 0])
lot_of_N2 = np.array([3., 0, 3, 0, 0, 0])
many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]
# Similarity
all_sents = np.exp(-np.array(all_sents))

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
#tau = 1.
ntsteps = 10000
noisemag = 0.001
nreps = 100

# For saving final states; dims: length, N1 Type, parse type(N1, N2, other)
data = np.zeros((len(pp), len(all_sents), 3))

for length in range(len(pp)):
    if length == 0:
        print('Starting -PP')
        # Half the boost if short
#        adj = 4.
        adj = 0.05
    else:
        print('Starting +PP')
#        adj = 2.
        adj = 0.1
        
    for sent in range(all_sents.shape[0]):
    # Set current input
        if sent == 3:
            all_sents[3,3] += np.random.uniform(0, 0.001, 1)
            all_sents[3,5] += np.random.uniform(0, 0.001, 1)
            ipt = all_sents[3,]
            x0 = np.array([0, 0, 0, 0.101, 0., 0.001])
        else:
            ipt = all_sents[sent] + np.random.uniform(0, 0.001, nlinks)
            x0 = np.array([0.001]*nlinks)
            x0[0] += 0.1
        print('\tStarting sentence {}'.format(sent))
    
        for rep in range(nreps):
        # For each repetition, reset history and noise
            xhist = np.zeros((ntsteps, nlinks))
            xhist[0,] = x0
#            noise = np.random.normal(0, noisemag, xhist.shape)
            noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)
        
            for t in range(1, ntsteps):
                # Euler forward dynamics
#                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#                * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
#                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#                * (ipt - W @ (ipt * xhist[t-1,]))) + noise[t,:], -0.01, 1.01)
                xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
                * (1 - W @ xhist[t-1,])) + noise[t,:], -0.01, 1.01)

                if sent != 3:
                    if t == 400:
                        xhist[t,1] += adj
                        xhist[t,2] += adj
                    if t == 800:
                        xhist[t,3:] += adj
                else:
                    xhist[t,0:3] = np.clip(noise[t,0:3], -0.1, 1.1)
                    xhist[t,4] = np.clip(noise[t,4], -0.01, 1.01)
                    if t == 400:
                        xhist[t,5] += adj

            # Tallying the final states        
            final = np.round(xhist[-1,])   
            if sent != 3:
                if np.all(final == [1, 0, 1, 0, 1, 0]):
                    data[length, sent, 0] += 1
                elif np.all(final == [0, 1, 0, 1, 0, 1]):
                    data[length, sent, 1] += 1
                else:
                    data[length, sent, 2] += 1
            else:
                if np.all(final == [1, 0, 1, 0, 1, 0]):
                    data[length, sent, 0] += 1
                elif np.all(final == [0, 0, 0, 1, 0, 1]):
                    data[length, sent, 1] += 1
                else:
                    data[length, sent, 2] += 1

data_scaled = data / nreps

for i in range(len(pp)):
    print('\n{}'.format(pp[i]))
    print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}\nQuantifiers:\t{}'.format(*data_scaled[i]))
    
for i in range(2):
    plt.plot(data_scaled[i,:, 1], 'o', label=pp[i])
plt.legend()
plt.title('Proportions of N2-headed parses')
plt.ylim(-0.05, 1.05)
plt.ylabel('Proportion N2')
plt.xticks([0, 1, 2, 3], ['Containers', 'Collections', 'Measures', 'Quantifiers'])
plt.show()
