# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:32:27 2017

@author: garrettsmith
"""

# Trying out simple treelet dynamics
# Goal: model Barker et al. (2001)'s Experiment 2 results:
# #pl verbs with/ 'canoe by sailboats' > 'canoe by cabins' due to semantic
# similarity between 'canoe' and 'sailboats.'

import numpy as np
import matplotlib.pyplot as plt

# Defining for later
def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Basics
nlinks = 3
nfeatures = 2
link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']
feat_names = ['Boat', 'Plural']

# Setting up integration
tau = 0.01
ntsteps = 1000
#tvec = np.arange(0, ntsteps, dtype=int)

# Setting up link variables
# Order is N1-V, N2-V, N2-ModN1
#k = 2
k = 0.75
x = np.zeros((ntsteps, nlinks))
#x[0,] = np.array([0.2]*nlinks)
x[0,] = np.array([0.001]*nlinks)
#x[0,] = np.array([0.01, 0.99, 0.01])
#adj = 2.
adj = 5.
# Setting first word to between its current state and 1
x[0,0] = x[0,0] + (1 - x[0,0]) / adj
Wx = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])

# Setting up feature vectors
fv = np.zeros((ntsteps, nfeatures))
#fv[0,] = np.random.uniform(0.47, 0.53, size=2)
fv[0,] = np.array([0.5]*nfeatures)
fn1 = np.array([1, 0])
# Cabins
cabins = np.array([0, 1])
# Sailboats
sailboats = np.array([1, 1])
#fn2n1 = cos_sim(fn2, np.array([0.1]*2))

# Noise
# noise = 0.5 seems too high
noise_mag = 0.1
x_noise = np.random.normal(0, noise_mag, size=x.shape)
fv_noise = np.random.normal(0, noise_mag, size=fv.shape)
clip = 0.1
#fv_noise = np.zeros(fv.shape)

ipt = np.zeros((ntsteps-1, nfeatures))
nruns = 100
corr_parse = np.array([1, 0, 1, 1, 0])
agr_attr = np.array([1, 0, 1, 1, 1])
corr_no_boat = np.array([1, 0, 1, 0, 0])
agr_attr_no_boat = np.array([1, 0, 1, 0, 1])
data = np.zeros((5, 2))
for sent in range(2):
    if sent == 0:
        fn2 = cabins
        print('Starting cabins')
    elif sent == 1:
        fn2 = sailboats
        print('Starting sailboats')
    fn2n1 = cos_sim(fn2, np.array([0.5]*nfeatures))
    for run in range(nruns):
        if run % 100 == 0:
            print('Run #{}'.format(run))
        x = np.zeros((ntsteps, nlinks))
#        x[0,] = np.array([0.2]*3)
        x[0,] = np.array([0.01]*3)
        x[0,0] = x[0,0] + (1 - x[0,0]) / adj
        fv = np.zeros((ntsteps, nfeatures))
        fv[0,] = np.array([0.5]*nfeatures)
        ipt = np.zeros((ntsteps-1, nfeatures))
        x_noise = np.random.normal(0, noise_mag, size=x.shape)
        fv_noise = np.random.normal(0, noise_mag, size=fv.shape)
        for t in range(1, ntsteps):
            prev_x = x[t-1,]
            prev_fv = fv[t-1,]
            m = [cos_sim(fn1, prev_fv), cos_sim(fn2, prev_fv), fn2n1]
            next_x = prev_x + tau * (m * prev_x * (1 - Wx @ prev_x) + x_noise[t-1,])
#            if t == 100:
            if t == 25:
                next_x = next_x + (1 - next_x)/adj
            x[t,] = np.clip(next_x, 0-clip, 1+clip)
            n1norm = np.linalg.norm(fn1)
            n2norm = np.linalg.norm(fn2)
            ipt[t-1,] = (prev_x[0] * (fn1/n1norm) + prev_x[1] * (fn2/n2norm)) #/ (prev_x[0] + prev_x[1])
            ipt[t-1,] = 2*ipt[t-1] - 1
#    ipt[t-1,0] = (prev_x[0]*fn1[0] + prev_x[1]*fn2[0])
#    ipt[t-1,1] = (prev_x[0]*fn1[1] + prev_x[1]*fn2[1])
#    next_fv = prev_fv + tau * (prev_fv * 
#      (ipt[t-1] - prev_fv) * (prev_fv - ipt[t-1]/2.) + fv_noise[t-1,])
#    next_fv = prev_fv + tau * ((2*ipt[t-1] - 1) * prev_fv * (1 - prev_fv) 
#      + fv_noise[t-1,])
#    next_fv = prev_fv + tau * (ipt[t-1] - prev_fv + fv_noise[t-1,])
#    next_fv = ipt[t-1,]
            next_fv = prev_fv + tau * (prev_fv * (1 - prev_fv) * (prev_fv - 0.5 + 0.01*ipt[t-1])
              + fv_noise[t-1,])
            fv[t,] = np.clip(next_fv, 0-clip, 1+clip)
#    fv[t,] = next_fv
        final = np.concatenate((x[-1,], fv[-1,]))
        final = np.round(final)
        if np.all(final == corr_parse):
            data[0, sent] += 1
        elif np.all(final == agr_attr):
            data[1, sent] += 1
        elif np.all(final == corr_no_boat):
            data[2, sent] += 1
        elif np.all(final == agr_attr_no_boat):
            data[3, sent] += 1
        else:
            data[4, sent] += 1
#        if final[-1] == 0:
#            data[0, sent] += 1
#        elif final[-1] == 1:
#            data[1, sent] += 1
#        else:
#            data[-1, sent] += 1
    
print('Correct: {}\nAgr. Attr.: {}\nCorrect, but no boat: {}\nAgr. Attr., no boat: {}\nOther: {}'.format(*data))
#print(data)
# Plotting trajectory
if nruns == 1:
    for traj in range(x.shape[1]):
        plt.plot(x[:,traj], label=link_names[traj])
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()

    for feat in range(fv.shape[1]):
        plt.plot(fv[:,feat], label=feat_names[feat])
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()

    plt.plot(ipt)
