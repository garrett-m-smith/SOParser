# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:34:11 2017

@author: garrettsmith
"""

# starting from scratch
# Without incorporating feature overlap at all, it gets more Agr.Attr. for
# 'sailboats', although both are about 50/50 (noisemag = 0.75)

import numpy as np
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

nlinks = 3
nfeatures = 2
nsents = 2

feat_names = ['Boat', 'Plural']
link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']

ntsteps = 1000
tau = 0.01

x = np.zeros((ntsteps, nlinks))
x[0,] = [0.2]*nlinks
adj = 2.
#x[0,0] = x[0,0] + (1 - x[0,0])/adj
k = 2.
W = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])
f = np.zeros((ntsteps, nfeatures))
f[0,] = [0.51, 0.49]
canoe = np.array([1, 0])
cabins = np.array([0, 1])
sailboats = np.array([1, 1])
n2s = np.array([cabins, sailboats])

lt = ['-', '--']

# Noise
# too little noise seems to make more Agr.Attr. in 'cabins'...
noise_mag = 0.0
clip = 0.1

data = np.zeros((3, nsents))
nrep = 10

m = np.zeros((ntsteps-1, 3))

for sent in range(nsents):
    print('Starting sentence {}'.format(sent))
    fn2 = n2s[sent,]
    for rep in range(nrep):
        if rep % 100 == 0:
            print('Repetition #: {}'.format(rep))
        x_noise = np.random.normal(0, noise_mag, size=x.shape)
        fv_noise = np.random.normal(0, noise_mag, size=f.shape)
        for t in range(1, ntsteps):
#        x[t,] = x[t-1,] + tau * (m[sent,] * x[t-1,] * (1 - W @ x[t-1,]))
            mn2 = cos_sim(fn2, f[t-1,])
            mn1 = cos_sim(canoe, f[t-1,])
            m[t-1,] = [mn1, mn2, 1.0]
            x[t,] = x[t-1,] + tau * (m[t-1,]*x[t-1,] * (1 - W @ x[t-1,]) + x_noise[t-1,])
#            if t==100:
#                x[t,1:] = x[t,1:] + (1 - x[t,1:])/adj
            x[t,] = np.clip(x[t,], 0-clip, 1+clip)
            fn2_norm = np.linalg.norm(fn2)
            f[t,] = f[t-1,] + tau * (f[t-1,] * (1 - f[t-1,])
                * (f[t-1,] - 0.5 + 0.1*(x[t-1,0]*canoe + x[t-1,1]*fn2/fn2_norm))
                + fv_noise[t-1,])
            f[t,] = np.clip(f[t,], 0-clip, 1+clip)
            
        final = np.round(np.concatenate((x[-1,], f[-1,])))
        if final[-1] == 0:
            data[0, sent] += 1
        elif final[-1] == 1:
            data[1, sent] += 1
        else:
            data[-1, sent] += 1

    if nrep == 1:
        plt.figure(1, figsize=(10, 7))
        plt.subplot(211)
        for c, l in zip(range(f.shape[1]), feat_names):
            plt.plot(f[:,0], f[:,1], linestyle=lt[sent])
            plt.legend()
            plt.ylim(0, 1)
            plt.xlim(0, 1)
#        plt.title('Agreement attraction attractor [1, 1]')
        
        plt.subplot(212)
        for c, l in zip(range(x.shape[1]), link_names):
            plt.plot(x[:,c], linestyle=lt[sent], label=l)
        plt.legend()
#        plt.subplot(313)
#        for c in range(m.shape[1]):
#            plt.plot(m[:,c], linestyle=lt[sent], label='match{}'.format(c))
#        plt.legend()
plt.title('Link trajectories')
plt.xlabel('Time')
plt.ylabel('Link strength')
plt.show()
print(data)
#for c, l in zip(range(x.shape[1]), feat_names):
#    plt.plot(f[:,c], linestyle=lt[sent], label=l)
#plt.legend(bbox_to_anchor=(0.95, 0.95))
#plt.show()

