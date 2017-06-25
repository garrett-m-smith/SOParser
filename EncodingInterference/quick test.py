# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:52:30 2017

@author: garrettsmith
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

link_names = ['N1->Subj(V)', 'N2->Subj(V)', 'N2->Mod(N1)']
feat_names = ['+/- boat', '+/- plural']

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

k = 1.0
W = np.array([[1, k, 0],
              [k, 1, k],
              [0, k, 1]])
#m_cabins = np.array([0.7, 0.1, 0.5]) #* 0.1
#m_sailboats = np.array([0.7, 0.7, 0.5]) #* 0.1

# Canoe
canoe = np.array([1, 0])
# Cabins
cabins = np.array([0, 1])
# Sailboats
sailboats = np.array([1, 1])
# NPmod
npmod = np.array([0.5]*2)

def dyn(x, t, n2):
    l = x[0:3]
    f = x[3:]
    m = [cos_sim(canoe, f), cos_sim(n2, f), cos_sim(n2, npmod)]
    dl = m * l * (1 - W @ l)
    net = f - 0.5 + 0.05*(l[0]*cabins + l[1]*n2)
    df = f * (1 - f) * net
    return np.concatenate((dl, df))

tvec = np.linspace(0, 40, 100)
#x0 = np.array([0.001]*3)
#x0 = np.array([0.01, 0.005, 0.005])
#x0 = np.array([0.001]*5)
x0 = np.array([0.001, 0.001, 0.001, 0.51, 0.49])

soln_cabins = odeint(dyn, x0, tvec, args=(cabins,))
soln_sailboats = odeint(dyn, x0, tvec, args=(sailboats,))

for i in range(3):
    plt.plot(soln_cabins[:,i], label=link_names[i])
    plt.plot(soln_sailboats[:,i], label=link_names[i], linestyle='--')
plt.legend()
plt.title("Link dynamics: dashed = 'sailboats'")
plt.xlabel('Time')
plt.ylabel('Link strength')
plt.show()

for i in range(2):
    plt.plot(soln_cabins[:,i+3], label=feat_names[i])
    plt.plot(soln_sailboats[:,i+3], label=feat_names[i], linestyle='--')
plt.legend()
plt.title("Feature dynamics: dashed = 'sailboats'")
plt.xlabel('Time')
plt.ylabel('Feature state')
plt.show()
