# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:17:13 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

nlinks = 3
nfeatures = 3
link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']
feat_names = ['Not-Boat', 'Boat', 'Plural']

# Setting up integration
tau = 0.01
ntsteps = 1000
#tvec = np.arange(0, ntsteps, dtype=int)

# Setting up link variables
# Order is N1-V, N2-V, N2-ModN1
k = 2
x = np.zeros((ntsteps, nlinks))
x[0,] = np.array([0.2]*nlinks)
#x[0,] = np.array([0.5]*nlinks)
#x[0,] = np.array([0.01, 0.99, 0.01])
adj = 2.
# Setting first word to between its current state and 1
x[0,0] = x[0,0] + (1 - x[0,0]) / adj
Wx = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])

# Setting up feature vectors
# Verb
fv = np.zeros((ntsteps, nfeatures))
# Verb starts at 0.5 on both features
fv[0,] = np.array([0.5]*nfeatures)
# Feature vector for n1; always 'canoe'
fn1 = np.array([0, 1, 0])
# Cabins: not boat-like and plural
cabins = np.array([1, 0, 1])
# Sailboats: boat-like and plural
sailboats = np.array([0, 1, 1]) #/ np.linalg.norm([1,1])

x = np.zeros((ntsteps, nlinks))
# All links start at 0.2
x[0,] = np.array([0.2]*nlinks)
# Giving boost to N1->V
x[0,0] = x[0,0] + (1 - x[0,0]) / adj
# Initializing verb features
fv = np.zeros((ntsteps, nfeatures))
fv[0,] = np.array([0.5]*nfeatures)
fn2n1 = 1.

ipt = np.zeros((ntsteps-1, nfeatures))
clip = 0.1

for sent in range(2):
    if sent==0:
        fn2 = cabins
        lt = '-'
    elif sent==1:
        fn2 = sailboats
        lt = '--'
    for t in range(1, ntsteps):
        prev_x = x[t-1,]
        prev_fv = fv[t-1,]
        m = [cos_sim(fn1, prev_fv), cos_sim(fn2, prev_fv), fn2n1]
#        m = [1., 1., 1.]
        next_x = prev_x + tau * (m * prev_x * (1 - Wx @ prev_x))
        if t == 100:
            next_x[1:] = next_x[1:] + (1 - next_x[1:])/adj
        x[t,] = np.clip(next_x, 0-clip, 1+clip)
        
        n1norm = np.linalg.norm(fn1)
        n2norm = np.linalg.norm(fn2)
        ipt[t-1,] = (prev_x[0]*(fn1/n1norm) + prev_x[1]*(fn2/n2norm)) #/ (prev_x[0] + prev_x[1])
#        ipt[t-1,] = prev_x[0]*fn1 + prev_x[1]*fn2
        ipt[t-1,] = 2*ipt[t-1,] - 1
        next_fv = prev_fv + tau * (prev_fv * (1 - prev_fv) * (prev_fv - 0.5 + 0.01*ipt[t-1,]))
        fv[t,] = np.clip(next_fv, 0-clip, 1+clip)
        
    for traj in range(x.shape[1]):
        plt.plot(x[:,traj], label=link_names[traj], linestyle=lt)
        plt.legend()
        plt.ylim(-0.1, 1.1)

    for feat in range(fv.shape[1]):
        plt.plot(fv[:,feat], label=feat_names[feat], linestyle=lt)
        plt.legend()
        plt.ylim(-0.1, 1.1)

plt.show()
