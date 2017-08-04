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
solns = np.load('../PsPartPaperModel/PsPartFixedPoints.npy')
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

# Trying a numerical method
# Issue: only approximations, so looking at eigenvals could be misleading...
#def sys_num(x):
#    k = 1.0
#    x0, x1, x2, x3, x4, x5 = x
#    dx0 = x0 * (1. - x0 - k * (x1 + x3 + x5))
#    dx1 = x1 * (1. - x1 - k * (x0 + x2 + x4))
#    dx2 = x2 * (1. - x2 - k * (x1 + x3 + x5))
#    dx3 = x3 * (1. - x3 - k * (x0 + x2 + x4))
#    dx4 = x4 * (1. - x4 - k * (x1 + x3 + x5))
#    dx5 = x5 * (1. - x5 - k * (x0 + x2 + x4))
#    return(dx0, dx1, dx2, dx3, dx4, dx5)
#
#solns = []
#for i in range(1000):
##    if i % 10 == 0: print(i)
#    rand_init = np.random.uniform(-0.5, 1.5, nlinks).tolist()
#    s = np.round(fsolve(sys_num, rand_init), 2)
#    if not any((s == x).all() for x in solns):
#        solns.append(s)
#
