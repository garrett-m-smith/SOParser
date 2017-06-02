# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:51:28 2017

@author: garrettsmith

Testing the equations for the treelet + activations system

Results: seems to behave as intended: rates of correct and incorrect parses
correlate with the relative feature overlap between 'these' and the nouns. 
There seem to be two 'other' parses: the origin, which is a non-hyperbolic
fixed point, possibly non-isolated, and [0, 1, 0, 0, 1].

At any rate, these are encouraging results. A caveat: all of the treelets are
introduced simulatneuosly; only the initial conditions are random. Introducing
the treelets one at a time will change things some, but this basic test is a 
good start.
"""

import numpy as np
from scipy.integrate import odeint
#import matplotlib.pyplot as plt


# The setup:
# These -l1-> dog vs. These -l2-> dogs
# Activations for 'these', 'dog', and 'dogs': 5 dim all together
# Need feature vectors & their overlaps

k = 2
feat_overlap = np.array([0.1, 0.9])
theta = 0.05
def dyn(x, t, k, feat_overlap, theta):
    dx = np.zeros(len(x))
    dx[0] = feat_overlap[0] * (x[2] + x[3]) * x[0] * (1 - x[0] - k * x[1])
    dx[1] = feat_overlap[1] * (x[2] + x[4]) * x[1] * (1 - x[1] - k * x[0])
    dx[2] = (0.5 * (x[0] + x[1]) - theta) * x[2] * (1 - x[2])
    dx[3] = (x[0] - theta) * x[3] * (1 - x[3])
    dx[4] = (x[1] - theta) * x[4] * (1 - x[4])
    return dx
    
#x0 = np.random.uniform(0.01, 0.1, size = 5)
tvec = np.linspace(0, 1000, 4000)
#soln = odeint(dyn, x0, tvec, args = (k, feat_overlap, theta))
#
#labels = ['l1', 'l2', 'a_these', 'a_dog', 'a_dogs']
#for s, l in zip(soln.T, labels):    
#    plt.plot(tvec, s, label = l)
#plt.legend()
#plt.show()

# Monte-Carlo runs with random initial conditions in [-0.1, 0.1]
nrep = 10000
correct_parse = np.array([0., 1, 1, 0, 1])
ncorrect = 0
incorrect_parse = np.array([1., 0, 1, 1, 0])
nincorrect = 0
nother = 0
for rep in range(nrep):
    if rep % 100 == 0:
        print('Repetition # {}'.format(rep))
    
    x0 = np.random.uniform(0.0, 0.1, 5)
    soln = odeint(dyn, x0, tvec, args = (k, feat_overlap, theta))
    if np.all(np.equal(np.round(soln[-1,:]), correct_parse)):
        ncorrect += 1
    elif np.all(np.equal(np.round(soln[-1,:]), incorrect_parse)):
        nincorrect += 1
    else:
        nother += 1
        print('Init: {}'.format(np.round(x0, 2)), 
        'Final: {}'.format(np.round(soln[-1,:], 2)))

print('Correct: {}\nIncorrect: {}\nOther: {}'.format(ncorrect, nincorrect, nother))