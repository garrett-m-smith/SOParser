# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:57:40 2017

@author: garrettsmith
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

nlinks = 6
# WTA weights
W = np.array([[1., 2, 0, 0, 0, 2], 
              [2, 1, 2, 0, 2, 0],
              [0, 2, 1, 2, 0, 0],
              [0, 0, 2, 1, 2, 0],
              [0, 2, 0, 2, 1, 2],
              [2, 0, 0, 0, 2, 1]])
link_names = ['N1->V', 'N1->P', 'P->N1', 'P->N2', 'N2->P', 'N2->V']

dx = np.zeros(nlinks)
def dyn(x, t):
    dx = x * (1 - W @ x)
    return dx

#x0 = np.random.uniform(0.1, 0.2, nlinks)
##x0 = np.array([0.05, 0.5, 0.15, 0.45, 0.1, 0.55])
#
#tvec = np.linspace(0, 15, 101)
#
#xx = odeint(dyn, x0, tvec)
#
#plt.figure()
#plt.ylim(-0.01, 1.01)
#for d in range(xx.shape[1]):
#    plt.plot(xx[:,d], label = link_names[d])
#    xpos = d * (len(xx[:,d]) / len(link_names))
#    ypos = xx[xpos, d]
#    plt.text(xpos, ypos, link_names[d])
#plt.legend()
#plt.show()
#
#for d in range(len(link_names)):
#    print('{}:\t{}'.format(link_names[d], np.round(xx[-1,d], 4)))

# Monte Carlo
nrep = 1000
n1headed = 0
n2headed = 0
other = 0
tvec = np.linspace(0, 15, 101)
for rep in range(nrep):
    x0 = np.random.uniform(0.0, 1.0, nlinks)
    xx = odeint(dyn, x0, tvec)
    final = np.round(xx[-1,], 2)
    if np.allclose(final, np.array([1., 0, 1, 0, 1, 0]), rtol = 0.1):
        n1headed += 1
    elif np.allclose(final, np.array([0., 1, 0, 1, 0, 1]), rtol = 0.1):
        n2headed += 1
    else:
        other += 1

print('N1 headed: {0}\nN2 headed: {1}\nOther: {2}'.format(n1headed, 
      n2headed, other))


