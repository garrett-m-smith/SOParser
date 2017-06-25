# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:39:02 2017

@author: garrettsmith
"""

# Third attempt
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:52:30 2017

@author: garrettsmith
"""

#from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']
feat_names = ['+/- boat', '+/- plural']

nlinks = 3
nfeat = 2

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

k = 1.0
W = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])

# Canoe
canoe = np.array([1, 0])
# Cabins
cabins = np.array([0, 1])
# Sailboats
sailboats = np.array([1, 1])
n2s = [cabins, sailboats]
# NPmod
npmod = np.array([0.5]*2)

def dyn(x, n2):
    l = x[0:3]
    f = x[3:]
    m = [cos_sim(canoe, f), cos_sim(n2, f), cos_sim(n2, npmod)]
    dl = np.clip(m * l * (1 - W @ l) 
                 + np.random.normal(0, 0.1, size=l.size), -0.1, 1.1)
#    net = (f - 0.5 + 0.001*(l[0]*cabins + l[1]*n2)
#            + np.random.normal(0, 0.1, size=f.size))
    net = 0.45*(l[0]*cabins + l[1]*n2) - f
    df = np.clip(f * (1 - f) * net 
                 + np.random.normal(0, 0.1, size=f.size), -0.1, 1.1)
#    df = np.clip(f * (1 - f) * net, -0.1, 1.1)
    return np.concatenate((dl, df))

tvec = np.linspace(0, 40, 100)
#x0 = np.array([0.001]*3)
#x0 = np.array([0.01, 0.005, 0.005])
#x0 = np.array([0.001]*5)
#x0 = np.array([0.001, 0.001, 0.001, 0.51, 0.49])
x0 = np.array([0.1, 0.001, 0.001, 0.5, 0.5])

tau = 0.01
ntsteps = 2000
x = np.zeros((ntsteps, nfeat+nlinks))
x[0,:] = x0
lines = ['-', '--']
nruns = 1
data = np.zeros((3, 2))
for sent in range(len(n2s)):
    print('Starting sentence {}'.format(sent))
    n2 = n2s[sent]
    ls = lines[sent]
    for run in range(nruns):
        if run % 50 == 0:
            print('Run {}'.format(run))
        for t in range(1, ntsteps):
            x[t,:] = x[t-1,:] + tau * dyn(x[t-1,:], n2)
            if t == 20:
                x[t,1:3] += 0.1
    
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
#    plt.show()
        final = np.round(x[-1,:])
        if final[-1] == 0:
            data[0,sent] += 1
        elif final[-1] == 1:
            data[1,sent] += 1
        else:
            data[2,sent] += 1

print('Singular agreement: {}\nPlural agreement: {}\nOther: {}'.format(*data))
