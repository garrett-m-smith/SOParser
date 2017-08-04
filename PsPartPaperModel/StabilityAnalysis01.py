# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:24:38 2017

@author: garrettsmith
"""

# Stability analysis using k as a control parameter

import numpy as np
from sympy import symbols, Matrix, re#, solve#, init_printing
from sympy.solvers.solveset import nonlinsolve
from sympy import solve
#from scipy.optimize import fsolve


nlinks = 6
x0, x1, x2, x3, x4, x5 = symbols('x:6', real=True)
k = symbols('k')
sys = Matrix([x0 * (1. - x0 - k * (x1 + x3 + x5)),
              x1 * (1. - x1 - k * (x0 + x2 + x4)),
              x2 * (1. - x2 - k * (x1 + x3 + x5)),
              x3 * (1. - x3 - k * (x0 + x2 + x4)),
              x4 * (1. - x4 - k * (x1 + x3 + x5)),
              x5 * (1. - x5 - k * (x0 + x2 + x4))])

jac = sys.jacobian([x0, x1, x2, x3, x4, x5])
#genera_solns = nonlinsolve(sys, [x0, x1, x2, x3, x4, x5]) # never finishes
#solns2 = solve(sys, [x0, x1, x2, x3, x4, x5], simplify=False) # Works
solns = np.load('PsPartFixedPoints.npy')
k_range = np.linspace(-2., 2., 200)

# Trying out a range of k values. Each list will hold the k values for which
# the relevant parse has the given stability
n1stable = []
n2stable = []
stable = [n1stable, n2stable]
n1saddle = []
n2saddle = []
saddle = [n1saddle, n2saddle]
n1unstable = []
n2unstable = []
unstable = [n1unstable, n2unstable]

n1 = {x0: 1, x1: 0, x2: 1, x3: 0, x4: 1, x5: 0}
n2 = {x0: 0, x1: 1, x2: 0, x3: 1, x4: 0, x5: 1}
parses = [n1, n2]

for i in range(len(k_range)):
    if i % 10 == 0: print('{}%'.format(i/len(k_range) * 100))
    for p in range(len(parses)):
        parses[p].update({k: k_range[i]})
        eig = jac.subs(parses[p]).eigenvals()
        vals = list(eig.keys())
        if all(re(v) < 0. for v in vals):
            stable[p].append(k_range[i])
        elif any(re(v) < 0. for v in vals) and any(re(v) > 0. for v in vals):
            saddle[p].append(k_range[i])
        else:
            unstable[p].append(k_range[i])

# Good news: the parse-fixed points undergo a bifurcation from saddles to 
# attractors at (I'm pretty sure) k = 1/3. Can't say much about the other
# fixed points, but this at least is good.

# Classifying fps for three values of k
dims = [x0, x1, x2, x3, x4, x5]
stable = []
saddles = []
unstable = []
nonhyp = []
ks = [0.5, 1.0, 2.0]
#ks = [-0.2]
for kval in range(len(ks)):
    for fp in range(solns.shape[0]):
        pts = [i.subs({k: ks[kval]}) for i in solns[fp,]]
        assign = dict(zip(dims, pts))
        assign[k] = ks[kval]
        eig = jac.subs(assign).eigenvals()
        vals = list(eig.keys())
        if all(re(i) < 0. for i in vals):
            print('k = {}, st. fp at {}'.format(ks[kval], pts))
        elif all(re(i) == 0. for i in vals):
            print('k = {}, nonhyperbolic fp at {}'.format(ks[kval], pts))


