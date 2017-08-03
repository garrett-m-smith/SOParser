# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
pp = ['+PP', '-PP']

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
#box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
#group_of_N2 = np.array([2., 2, 2, 2, 0, 0])
#lot_of_N2 = np.array([3., 0, 3, 0, 0, 0])
#many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

#box_of_N2 = np.array([0., 3, 0, 3, 0, 0])
#group_of_N2 = np.array([2., 2, 2, 2, 2, 0])
#lot_of_N2 = np.array([3., 0, 3, 0, 3, 0])
#many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

#box_of_N2 = np.array([0., 1, 1, 1, 1, 0])
#group_of_N2 = np.array([2., 1, 1, 1, 1, 0])
#lot_of_N2 = np.array([3., 1, 1, 1, 1, 0])
#many_N2 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

box_of_N2_pp = np.array([0., 1, 1, 1, 1, 0])
group_of_N2_pp = np.array([1., 1, 1, 1, 1, 0])
lot_of_N2_pp = np.array([3., 1, 1, 1, 1, 0])
many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

box_of_N2_no = np.array([0., 2, 2, 2, 2, 1])
group_of_N2_no = np.array([1., 2, 2, 2, 2, 1])
lot_of_N2_no = np.array([3., 2, 2, 2, 2, 1])
many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])

#box_of_N2_pp = np.array([0., 3, 0, 3, 0, 0])
#group_of_N2_pp = np.array([1., 1, 1, 1, 1, 0])
#lot_of_N2_pp = np.array([3., 0, 3, 0, 3, 0])
#many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])
#
#box_of_N2_no = np.array([0., 4, 1, 4, 1, 1])
#group_of_N2_no = np.array([1., 2, 2, 2, 2, 1])
#lot_of_N2_no = np.array([3., 1, 4, 1, 4, 1])
#many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])

all_sents = [box_of_N2_pp, group_of_N2_pp, lot_of_N2_pp, many_N2_pp, box_of_N2_no, group_of_N2_no, lot_of_N2_no, many_N2_no]
#all_sents = np.array(all_sents)
# Similarity
#all_sents = normalize(np.exp(-np.array(all_sents)), norm='l2', axis=1)
all_sents = np.exp(-np.array(all_sents))

# Interaction matrix: specifies which links enter into WTA competitions. The
# parameter k determines the relative strength of inhibition from other links
# to a link's self-inhibition
k = 0.5 # gets even better fit. Need to look at individual runs, though...
#k = 1.01
# and also do a stability analysis to see if things still look the way they should
#k = 1.1 # works
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

## Monte Carlo
tau = 0.01
ntsteps = 10000
noisemag = 0.001 # works!
#nreps = 2000
nreps = 100
adj = 0.1

# For saving final states; dims: length, N1 Type, parse type(N1, N2, other)
data = np.zeros((len(all_sents), 3))

#for length in range(len(pp)):
#    if length == 0:
#        print('Starting -PP')
#         Half the boost if short
#        adj = 0.05
#        adj = adj0/2.
#        adj = 0.0
#    else:
#        print('Starting +PP')
#        adj = 0.1
#        adj = adj0
        
for sent in range(all_sents.shape[0]):
#for sent in range(7, 8):
    # Set current input
    ipt = all_sents[sent,]
    print('\tStarting sentence {}'.format(sent))
    
    for rep in range(nreps):
        # For each repetition, reset history and noise
        if sent == 3 or sent == 7:
            # Minus to keep < 1
#            all_sents[3,3] -= np.random.uniform(0, 0.001, 1)
#            all_sents[3,5] -= np.random.uniform(0, 0.001, 1)
#            ipt = all_sents[3,]
            x0 = np.array([0, 0, 0, 0.101, 0., 0.001])
#               x0 = np.array([0, 0, 0, 0.011, 0., 0.001])
        else:
#            ipt = all_sents[sent,] #+ np.random.uniform(0, 0.001, nlinks)
            x0 = np.array([0.001]*nlinks)
            x0[0] += 0.1
#               x0[0] += 0.01
        xhist = np.zeros((ntsteps, nlinks))
        xhist[0,] = x0
        noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)
            
        t = 0
#        while True:
        while t < ntsteps-1:
            t += 1
#            for t in range(1, ntsteps):
                # Euler forward dynamics
#                xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
#                * (ipt - W @ (ipt * xhist[t-1,]))) + noise[t,:], -0.01, 1.01)
            xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
            * (1 - W @ xhist[t-1,])) + noise[t-1,:], -0.01, 1.01)

#            if sent != 3 and sent != 7:
            if sent < 3:
                if t == 400:
                    xhist[t,1] += adj
                    xhist[t,2] += adj
                if t == 800:
                    xhist[t,3:] += adj
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
#                xhist[t,0:3] = np.clip(noise[t,0:3], -0.01, 1.01)
#                xhist[t,4] = np.clip(noise[t,4], -0.01, 1.01)
                if t == 400:
                    xhist[t,5] += adj
            
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
#                if t == 400:
#                    xhist[t,1:] += adj
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
#                if t == 400:
#                    xhist[t,-1] += adj
                xhist[t, 0:3] = 0
                xhist[t, 4] = 0
#                xhist[t,0:3] = np.clip(noise[t,0:3], -0.01, 1.01)
#                xhist[t,4] = np.clip(noise[t,4], -0.01, 1.01)
#                if t == 400:
#                    xhist[t,5] += adj
            
#                if t >= 800:
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

        # Tallying the final states
#        final = np.round(xhist[-1,])   
#        if sent == 3 or sent == 7:
#            if np.all(final == [1, 0, 1, 0, 1, 0]):
#                data[sent, 0] += 1
#            elif np.all(final == [0, 0, 0, 1, 0, 1]):
#                data[sent, 1] += 1
#            else:
#                data[sent, 2] += 1
#        else:
#            if np.all(final == [1, 0, 1, 0, 1, 0]):
#                data[sent, 0] += 1
#            elif np.all(final == [0, 1, 0, 1, 0, 1]):
#                data[sent, 1] += 1
#            else:
#                data[sent, 2] += 1

data_scaled = data / nreps

#for i in range(len(pp)):
print('\n{}'.format(pp[0]))
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}\nQuantifiers:\t{}'.format(*data_scaled[:4,]))
print('\n{}'.format(pp[1]))
print('Containers:\t{}\nCollections:\t{}\nMeasures:\t{}\nQuantifiers:\t{}'.format(*data_scaled[4:,]))
    
# Human data
human_data = np.array([[137, 97, 0],
                       [90, 145, 0],
                       [34, 201, 0],
                       [9, 227, 0],
                       [222, 14, 0],
                       [189, 46, 0],
                       [99, 134, 0],
                       [27, 206, 0]])
human_data = human_data / human_data.sum(axis=1)[:,None]

plt.figure(figsize=(10,6))
plt.plot(data_scaled[0:4, 1], 'b^', label=pp[0]+' Model')
plt.plot(data_scaled[4:, 1], 'bo', label=pp[1]+' Model')
plt.plot([0.1, 1.1, 2.1, 3.1], human_data[0:4, 1], 'y^', label=pp[0]+' Human')
plt.plot([0.1, 1.1, 2.1, 3.1], human_data[4:, 1], 'yo', label=pp[1]+' Human')
plt.legend()
plt.title('Proportions of N2-headed parses')
plt.ylim(-0.05, 1.05)
plt.ylabel('Proportion N2')
plt.xticks([0, 1, 2, 3], ['Containers', 'Collections', 'Measures', 'Quantifiers'])
plt.show()
