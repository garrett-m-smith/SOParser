# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:54:14 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
pp = ['+PP', '-PP']

# Setting the LV growth rates to plausible values given our feature cline.
# Each dimension corresponds to the links in link_labels above.
# For publication (Hamming distances between feature vectors for ea. link):
box_of_N2_pp = np.array([0., 0, 1, 1, 0, 0])
group_of_N2_pp = np.array([1., 0, 1, 1, 0, 0])
lot_of_N2_pp = np.array([3., 0, 1, 1, 0, 0])
many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])

box_of_N2_no = np.array([0., 1, 2, 2, 1, 1])
group_of_N2_no = np.array([1., 1, 2, 2, 1, 1])
lot_of_N2_no = np.array([3., 1, 2, 2, 1, 1])
many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])

all_sents = [box_of_N2_pp, group_of_N2_pp, lot_of_N2_pp, many_N2_pp, box_of_N2_no, group_of_N2_no, lot_of_N2_no, many_N2_no]
all_sents = np.exp(-np.array(all_sents))

# Interaction matrix: specifies which links enter into WTA competitions. The
# parameter k determines the relative strength of inhibition from other links
# to a link's self-inhibition
k = 1.01  # for publication
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

# Monte Carlo
tau = 0.01
ntsteps = 30000
noisemag = 0.001  # works! for publication
nreps = 1000  # for publication
#nreps = 50
adj = 0.1
isi = 100  # for publication

# For saving final states; dims: length, N1 Type, parse type(N1, N2, other)
data = np.zeros((len(all_sents), 3))

for sent in range(all_sents.shape[0]):
    ipt = all_sents[sent,]
    print('\tStarting sentence {}'.format(sent))

    for rep in range(nreps):
        # For each repetition, reset history and noise
        if sent == 3 or sent == 7:
            x0 = np.array([0, 0, 0, 0.101, 0., 0.001])
        else:
            x0 = np.array([0.001]*nlinks)
            x0[0] += 0.1
        xhist = np.zeros((ntsteps, nlinks))
        xhist[0,] = x0
        noise = np.sqrt(tau*noisemag) * np.random.normal(0, 1, xhist.shape)

        t = 0
        while t < ntsteps-1:
            t += 1
            # Euler forward dynamics
            xhist[t,] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
            * (1 - W @ xhist[t-1,])) + noise[t-1,], -0.01, 1.01)
            
            test_n1 = ((xhist[t, 0] > 0.5 > xhist[t, -1])
                       or (xhist[t, 1] > 0.5 > xhist[t, -2])
                       or (xhist[t, 2] > 0.5 > xhist[t, -3]))
            test_n2 = ((xhist[t, -1] > 0.5 > xhist[t, 0])
                       or (xhist[t, -2] > 0.5 > xhist[t, 1])
                       or (xhist[t, -3] > 0.5 > xhist[t, 2]))

            if sent < 3:
                if t == isi:
                    xhist[t, 1] += adj
                    xhist[t, 2] += adj
                if t == 2*isi:
                    xhist[t, 3:] += adj
                if t >= 3*isi:
                    if xhist[t, 0] > 0.5 and xhist[t, -1] < 0.5:
#                    if test_n1:
                        data[sent, 0] += 1
                        break
                    elif xhist[t, 0] < 0.5 and xhist[t, -1] > 0.5:
#                    elif test_n2:
                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
                        data[sent, 2] += 1
                        break
            elif sent == 3:
                xhist[t, 0:3] = 0
                xhist[t, 4] = 0
                if t == isi:
                    xhist[t, 5] += adj

                if t >= 2*isi:
                    if xhist[t, 0] > 0.5 and xhist[t, -1] < 0.5:
#                    if xhist[t, 0] > xhist[t, -1]:
                        data[sent, 0] += 1
                        break
                    elif xhist[t, 0] < 0.5 and xhist[t, -1] > 0.5:
#                    elif xhist[t, -1] > xhist[t, 0]:
                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
                        data[sent, 2] += 1
                        break
            elif sent > 3 and sent < 7:
                if t > isi:
                    if xhist[t, 0] > 0.5 and xhist[t, -1] < 0.5:
#                    if test_n1:
                        data[sent, 0] += 1
                        break
                    elif xhist[t, 0] < 0.5 and xhist[t, -1] > 0.5:
#                    elif test_n2:
                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
                        data[sent, 2] += 1
                        break
            else:
                xhist[t, 0:3] = 0
                xhist[t, 4] = 0

                if t > isi:
                    if xhist[t, 0] > 0.5 and xhist[t, -1] < 0.5:
#                    if xhist[t, 0] > xhist[t, -1]:
                        data[sent, 0] += 1
                        break
                    elif xhist[t, 0] < 0.5 and xhist[t, -1] > 0.5:
#                    elif xhist[t, -1] > xhist[t, 0]:
                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
                        data[sent, 2] += 1
                        break


data_scaled = data / nreps

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
human_data = human_data / human_data.sum(axis=1)[:, None]

plt.figure(figsize=(10, 6))
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

#np.save('ModelDataForPublication', data_scaled)
