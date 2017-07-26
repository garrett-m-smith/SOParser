# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:09:41 2017

@author: garrettsmith
"""

# Using SymPy to find all fixed points of the CUNY system.
# Later: compare to behavior of model from MC sims.

import numpy as np
from sympy import symbols, solve, Matrix, re#, init_printing

# Turning on pretty printing
#init_printing()


# State variables
x0, x1, x2, x3, x4, x5 = symbols('x:6', nonnegative=True)
#x = Matrix((x0, x1, x2, x3, x4, x5))

# Finding fixed points of nonlinear system
# box: [0.9, 0.3, 0.9, 0.9, 0.9, 0.9]
sys = Matrix([x0 * (0.9 - 0.9*x0 - 2 * (0.3*x1 + 0.9*x3 + 0.9*x5)),
       x1 * (0.3 - 0.3*x1 - 2 * (0.9*x0 + 0.9*x2 + 0.9*x4)),
       x2 * (0.9 - 0.9*x2 - 2 * (0.3*x1 + 0.9*x3 + 0.9*x5)),
       x3 * (0.9 - 0.9*x3 - 2 * (0.9*x0 + 0.9*x2 + 0.9*x4)),
       x4 * (0.9 - 0.9*x4 - 2 * (0.3*x1 + 0.9*x3 + 0.9*x5)),
       x5 * (0.9 - 0.9*x5 - 2 * (0.9*x0 + 0.9*x2 + 0.9*x4))])
fps = solve(sys, x0, x1, x2, x3, x4, x5, dict=True)
list_fps = [list(x.values()) for x in fps]
unique_fps = np.array([list(x) for x in set(tuple(x) for x in list_fps)])

# Linearizing
jac = sys.jacobian((x0, x1, x2, x3, x4, x5))

# Classifying fps
stable = []
saddles = []
unstable = []
for fp in fps:
    eig = jac.subs(fp).eigenvals()
    vals = list(eig.keys())
    if all(re(i) < 0. for i in vals):
        stable.append(fp)
    elif any(re(i) < 0. for i in vals) and any(re(i) > 0. for i in vals):
        saddles.append(fp)
    else:
        unstable.append(fp)
saddles_np = [list(x.values()) for x in saddles]
saddles_np = np.round(np.array(saddles_np, dtype=float),4)

# Now, getting other parses from the simulation:
# Monte Carlo
nlinks = 6
k = 2.
W = np.array([[1, k, 0, k, 0, k],
              [k, 1, k, 0, k, 0],
              [0, k, 1, k, 0, k],
              [k, 0, k, 1, k, 0],
              [0, k, 0, k, 1, k],
              [k, 0, k, 0, k, 1]])
box_of_N2 = np.array([0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
group_of_N2 = np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
lot_of_N2 = np.array([0.3, 0.9, 0.3, 0.9, 0.9, 0.9])
tau = 0.01
nsec = 50
tvec = np.linspace(0, nsec, nsec/tau + 1)
x0 = np.array([0.2] * nlinks)
adj = 2.
# Setting first word to between its current state and 1
x0[0] = x0[0] + (1 - x0[0]) / adj
xhist = np.zeros((len(tvec), nlinks))
xhist[0,] = x0
#noisemag = 1. # good!
#noisemag = 2. # better balance, but with more Others
noisemag = 1.5
noise = np.random.normal(0, noisemag, xhist.shape)

nreps = 1000
all_sents = [box_of_N2, group_of_N2, lot_of_N2]
data = np.zeros((len(all_sents), 3))
other_parses = np.zeros(nlinks)
for sent in range(len(all_sents)):
    ipt = all_sents[sent]
    x0[0] = ipt[0]
    print('Starting sentence {}'.format(sent))
    for rep in range(nreps):
        xhist = np.zeros((len(tvec), nlinks))
        xhist[0,] = x0
#        xhist[0,] = np.random.uniform(0.1, 0.2, xhist.shape[1])
        noise = np.random.normal(0, noisemag, xhist.shape)
        
        for t in range(1, len(tvec)):
            xhist[t,:] = np.clip(xhist[t-1,] + tau * (xhist[t-1,] 
            * (ipt - W @ (ipt * xhist[t-1,])) + noise[t,:]), -0.1, 1.1)
    
            if t == 100:
               # Turn boost P-P(N1) = intro Prep
                xhist[t,2] = xhist[t,2] + (1 - xhist[t,2]) / adj
                # Also turn on its competitor: N1-N(P)
                xhist[t,1] = xhist[t,1] + (1 - xhist[t,1]) / adj
            if t == 200:
                # Intro N2-related links: P-Det(N2)
                xhist[t,3] = xhist[t,3] + (1 - xhist[t,3]) / adj
                # N2-N(P)
                xhist[t,4] = xhist[t,4] + (1 - xhist[t,4]) / adj
                # N2-N(V)
                xhist[t,5] = xhist[t,5] + (1 - xhist[t,5]) / adj
        
        final = np.round(xhist[-1,])
        
#        if np.allclose(final, np.array([1., 0, 1, 0, 1, 0]), rtol = 0.4):
        if np.all(final == [1, 0, 1, 0, 1, 0]):
            data[sent,0] += 1
        elif np.all(final == [0., 1, 0, 1, 0, 1]):
#        elif np.allclose(final, np.array([0., 1, 0, 1, 0, 1]), rtol = 0.4):
            data[sent,1] += 1
        else:
            data[sent,2] += 1
            other_parses = np.vstack((other_parses, np.round(xhist[-1,], 4)))

print(data)
print(other_parses)

# Comparing the fps to the other_parses
for s in range(saddles_np.shape[0]):
    saddle = np.round(saddles_np[s,], 2)
    for p in range(other_parses.shape[0]):
        other = np.round(saddles_np[p,], 2)
        if np.all(other == saddle):
            print('Other:\t{}\nSaddle:\t{}'.format(other, saddle))
