# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:40 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize

nlinks = 6
link_names = ['N1-Verb', 'N1-of', 'of-N1', 'of-N2', 'N2-of', 'N2-V']
pp = ['+PP', '-PP']

#box_of_N2_pp = np.array([0., 1, 1, 1, 1, 0])
#group_of_N2_pp = np.array([1., 1, 1, 1, 1, 0])
#lot_of_N2_pp = np.array([3., 1, 1, 1, 1, 0])
#many_N2_pp = np.array([np.inf, np.inf, np.inf, 0, np.inf, 0])
#
#box_of_N2_no = np.array([0., 2, 2, 2, 2, 1])
#group_of_N2_no = np.array([1., 2, 2, 2, 2, 1])
#lot_of_N2_no = np.array([3., 2, 2, 2, 2, 1])
#many_N2_no = np.array([np.inf, np.inf, np.inf, 0, np.inf, 1])

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

k = 1.01
#k = 0.5
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])

## Monte Carlo
tau = 0.01
ntsteps = 30000
noisemag = 0.001
nreps = 200
adj = 0.1
isi = 100
        
#for sent in range(all_sents.shape[0]):
for sent in [6]:
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
#        while True:
        while t < ntsteps-1:
            t += 1
#            xhist[t,:] = np.clip(xhist[t-1,] + tau * (ipt * xhist[t-1,] 
#            * (1 - W @ xhist[t-1,])) + noise[t-1,:], -0.01, 1.01)
            xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
            * (ipt - W @ (ipt * xhist[t-1,]))) + noise[t-1,:], -0.01, 1.01)

#            if sent != 3 and sent != 7:
            if sent < 3:
                if t == isi:
                    xhist[t,1] += adj
                    xhist[t,2] += adj
                if t == 2*isi:
                    xhist[t,3:] += adj
                if t >= 3*isi:
                    if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
#                        data[sent, 0] += 1
                        break
                    elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
#                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
#                        data[sent, 2] += 1
                        break
            elif sent == 3:
                xhist[t, 0:3] = 0
                xhist[t, 4] = 0
                if t == isi:
                    xhist[t,5] += adj
            
                if t >= 2*isi:
                    if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
#                        data[sent, 0] += 1
                        break
                    elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
#                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
#                        data[sent, 2] += 1
                        break
            elif sent > 3 and sent < 7:
                # Assuming the elided material all comes in at once
#                if t == isi:
#                    xhist[t,1:] += adj
                if t > isi:
                    if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
#                        data[sent, 0] += 1
                        break
                    elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
#                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
#                        data[sent, 2] += 1
                        break
            else:
#                if t == 400:
#                    xhist[t,-1] += adj
                xhist[t, 0:3] = 0
                xhist[t, 4] = 0
                if t > isi:
                    if xhist[t,0] > 0.5 and xhist[t,-1] < 0.5:
#                        data[sent, 0] += 1
                        break
                    elif xhist[t,0] < 0.5 and xhist[t,-1] > 0.5:
#                        data[sent, 1] += 1
                        break
                    elif (t+1) == ntsteps:
#                        data[sent, 2] += 1
                        break


# If you want to save individual trajectories as CSVs
#np.savetxt('box-N1-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('group-N1-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('group-N2-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')
#np.savetxt('lot-N2-headed.csv', xhist.T, delimiter = ',', fmt = '%6f')

xhist = xhist[~np.all(xhist == 0, axis=1)]
# Printing out the final vlaues of each link
for d in range(len(link_names)):
    print('{}:\t{}'.format(link_names[d], np.round(xhist[-1,d], 4)))

plt.figure()
plt.ylim(-0.1, 1.1)
for d in range(xhist.shape[1]):
    plt.plot(xhist[:,d], label = link_names[d])
#    xpos = d * (len(xhist[:,d]) / len(link_names))
#    ypos = xhist[xpos, d]
#    plt.text(xpos, ypos, link_names[d])
plt.legend()
plt.title('Individual link competition run')
plt.show()

