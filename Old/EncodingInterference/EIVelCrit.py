# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:39:02 2017

@author: garrettsmith
"""


import numpy as np
import matplotlib.pyplot as plt


# Should probably switch to exp(-distance) similarity...
def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# The equations
def dyn(x, n2):
    """Takes a vector of link strengths and verb feature values and returns
    the same after updating according to the equations for the system."""
    # Separating the input vector into links and features
    l = x[0:3]
    f = x[3:]
    # Calculating the feature match
    m = [cos_sim(canoe, f), cos_sim(n2, f), cos_sim(n2, npmod)]
    dl = np.clip(m * l * (1 - W @ l)
                 + np.random.normal(0, 0.1, size=l.size), -0.1, 1.1)
    net = f - 0.5 + 0.01 * (l[0]*(2*canoe-1) + l[1]*(2*n2-1))
    # Note the difference in noise magnitues between links and features
    df = np.clip(f * (1 - f) * net
                 + np.random.normal(0, 0.01, size=f.size), -0.1, 1.1)
    return np.concatenate((dl, df))


def vel_threshold(x, threshold, tol):
    # Need to integrate feature match...
    vel = x * (1 - W @ x)
    mag = np.sqrt(vel @ vel)
    if np.abs(mag - threshold) > tol:
        return True
    else:
        return False


link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']
feat_names = ['+/- boat', '+/- plural']

nlinks = 3
nfeat = 2

# Competition parameter
k = 1.0
# Interaction matrix
W = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])

# Canoe mother features
canoe = np.array([1, 0])
# Cabins mother features
cabins = np.array([0, 1])
# Sailboats mother features
sailboats = np.array([1, 1])
n2s = [cabins, sailboats]
# NPmod features
npmod = np.array([0.5]*2)

# Initial conditions
#x0 = np.array([0.001]*3)
#x0 = np.array([0.01, 0.005, 0.005])
#x0 = np.array([0.001]*5)
#x0 = np.array([0.001, 0.001, 0.001, 0.51, 0.49])
x0 = np.array([0.01, 0.001, 0.001, 0.5, 0.5])

# Time constant and length of integration
tau = 0.01
ntsteps = 2000
x = np.zeros((ntsteps, nfeat+nlinks))
x[0, :] = x0
tol = 0.001
threshold = 0.1
isi = 250

# For plotting if nruns == 1
lines = ['-', '--']
nruns = 100
# To save the results
data = np.zeros((3, 2))
rts = np.zeros((nruns, 2))

for sent in range(len(n2s)):
    print('Starting sentence {}'.format(sent))
    n2 = n2s[sent]
    ls = lines[sent]
    for run in range(nruns):
        x = np.zeros((ntsteps, nfeat+nlinks))
        x[0, :] = x0
        if run % 50 == 0:
            print('Run {}'.format(run))
#        for t in range(1, ntsteps):
        t = 1
        while t < ntsteps:
            x[t,:] = x[t-1,:] + tau * dyn(x[t-1, :], n2)
            # Boost the links from N2 to the verb and NPmod
            if t == isi:
                x[t,1:3] += 0.01
            elif t >= 2*isi and not vel_threshold(x[t-1, 0:3], threshold, tol):
                break
            t += 1
        rts[run, sent] = t - 2*isi
    
        # Plotting individual runs
        if nruns == 1:
            plt.figure(1)
            for i in range(nlinks):
                plt.plot(x[:,i], label=link_names[i], linestyle=ls)
            plt.legend()
            plt.title("Link dynamics: dashed = 'sailboats'")
            plt.xlabel('Time')
            plt.ylabel('Link strength')

            plt.figure(2)
            for i in range(nfeat):
                plt.plot(x[:,i+3], label=feat_names[i], linestyle=ls)
            plt.legend()
            plt.title("Feature dynamics: dashed = 'sailboats'")
            plt.xlabel('Time')
            plt.ylabel('Feature state')

        final = x[np.all(x != 0, axis=1)].copy()
        final = np.round(final[-1,:])
        if final[-1] == 0:
            data[0,sent] += 1
        elif final[-1] == 1:
            data[1,sent] += 1
        elif final[1] != 0:
            data[2,sent] += 1


print('Singular agreement: {}\nPlural agreement: {}\nOther: {}\n'.format(
        *data))
print('Mean RT (cabins): {}\nMean RT (sailboats): {}'.format(
        *np.mean(rts, axis=1)))
