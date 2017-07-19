# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
pp = ['-PP', '+PP']

# Setting the LV growth rates to plausible values given our feature cline.
# Each dimension corresponds to the links in link_labels above.
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
many_N2 = np.array([0, 0, 0, 0.9, 0., 0.9])

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

nreps = 100
all_sents = [box_of_N2, group_of_N2, lot_of_N2, many_N2]

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
        if ipt is many_N2:
            x0 = np.array([0, 0, 0, 0.10, 0.001, 0.001])
        print('\tStarting sentence {}'.format(sent))
    
        for rep in range(nreps):
        # For each repetition, reset history and noise
            xhist = np.zeros((ntsteps, nlinks))
            xhist[0,] = x0
            noise = np.random.normal(0, noisemag, xhist.shape)
        
            for t in range(1, ntsteps):
                # Euler forward dynamics
                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
                * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)

                if ipt is not many_N2:
                    if t == 25:
                        xhist[t,2] += adj
                        xhist[t,1] += adj
                    if t == 50:
                        xhist[t,3:] += adj
                else:
                    xhist[t,0:3] = np.clip(0 + tau*noise[t,0:3], -0.1, 1.1)
                    if t == 25:
                        xhist[t,4:] += adj

            # Tallying the final states        
            final = np.round(xhist[-1,])   
            if ipt is not many_N2:
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

print('-PP:')
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}\nQuantifiers:\t{}'.format(*data_scaled[0]))
print('+PP:')
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}\nQuantifiers:\t{}'.format(*data_scaled[1]))
    
for i in range(2):
    plt.plot(data_scaled[i,:, 1], 'o', label=pp[i])
plt.legend()
plt.title('Proportions of N2-headed parses')
plt.ylim(-0.05, 1.05)
plt.ylabel('Proportion N2')
plt.xticks([0, 1, 2, 3], ['Containers', 'Collections', 'Measures', 'Quantifiers'])
plt.show()
