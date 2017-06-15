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
ntsteps = 3000
#tvec = np.arange(0, ntsteps, dtype=int)

# Setting up link variables
# Order is N1-V, N2-V, N2-ModN1
k = 1.5
x = np.zeros((ntsteps, nlinks))
x[0,] = np.array([0.05]*3)
Wx = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])

# Setting up feature vectors
fv = np.zeros((ntsteps, nfeatures))
#fv[0,] = np.random.uniform(0.47, 0.53, size=2)
fv[0,] = np.array([0.5]*2)
fn1 = np.array([1, 0])
fn2 = np.array([1, 1])
fn2n1 = cos_sim(fn2, np.array([0.1]*2))

# Noise
noise_mag = 0.
x_noise = np.random.normal(0, noise_mag, size=x.shape)
fv_noise = np.random.normal(0, noise_mag, size=fv.shape)
#fv_noise = np.zeros(fv.shape)

ipt = np.zeros((ntsteps-1, nfeatures))

for t in range(1, ntsteps):
    prev_x = x[t-1,]
    prev_fv = fv[t-1,]
    m = [cos_sim(fn1, prev_fv), cos_sim(fn2, prev_fv), fn2n1]
    next_x = prev_x + tau * (m * prev_x * (1 - Wx @ prev_x) + x_noise[t-1,])
    x[t,] = np.clip(next_x, -0.01, 1.01)
    if t == 200:
        x[t,0] += 0.2
    if t == 400:
        x[t,1:] += 0.2
    # Here is the problem: something about this is not right...
    # Boat feat should always go to 1...
    ipt[t-1] = (prev_x[0]*fn1 + prev_x[1]*fn2) #/ (prev_x[0] + prev_x[1])
    next_fv = prev_fv + tau * (prev_fv * 
      (ipt[t-1] - prev_fv) * (prev_fv - ipt[t-1]/2.) + fv_noise[t-1,])
#    next_fv = prev_fv + tau * ((2*ipt[t-1] - 1) * prev_fv * (1 - prev_fv) 
#      + fv_noise[t-1,])
    fv[t,] = np.clip(next_fv, -0.01, 1.01)
    
    
# Plotting trajectory
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
