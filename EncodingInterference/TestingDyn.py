# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:20:14 2017

@author: garrettsmith
"""

# Exploring eignvals from encoding interference model

from sympy import *
import numpy as np

#init_session()

def cos_sim(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Variables
cx, cy, cz, cf0, cf1 = symbols('cx, cy, cz, cf0, cf1')
sx, sy, sz, sf0, sf1 = symbols('sx, sy, sz, sf0, sf1')

# Features
canoe = np.array([1, 0.])
cabins = np.array([0, 1.])
sailboats = np.array([1, 1.])
sailboats_norm = np.linalg.norm(sailboats)
npmod = np.array([0.5, 0.5])
verb0 = np.array([0.5, 0.5])

cfz = cos_sim(npmod, cabins)
sfz = cos_sim(npmod, sailboats)


# Systems
sys_c = Matrix([(canoe[0]*cf0 + canoe[1]*cf1) * cx * (1 - cx - 2*cy),
                (cabins[0]*cf0 + cabins[1]*cf1) * cy * (1 - cy - 2*cx - 2*cz),
                cfz * cz * (1 - cz - 2*cy),
                cf0 * (1 - cf0) * (cf0 - 0.5 + 0.05 * (cx*canoe[0] + cy*cabins[0])),
                cf1 * (1 - cf1) * (cf1 - 0.5 + 0.05 * (cx*canoe[1] + cy*cabins[1]))])

sys_s = Matrix([(canoe[0]*sf0 + canoe[1]*sf1) * sx * (1 - sx - 2*sy),
                (sailboats[0]*sf0 + sailboats[1]*sf1) * sy * (1 - sy - 2*sx - 2*sz),
                sfz * sz * (1 - sz - 2*sy),
                sf0 * (1 - sf0) * (sf0 - 0.5 + 0.05 * (sx*canoe[0] + sy*sailboats[0]/sailboats_norm)),
                sf1 * (1 - sf1) * (sf1 - 0.5 + 0.05 * (sx*canoe[1] + sy*sailboats[1]/sailboats_norm))])

sys_alt = Matrix([(canoe[0]*cf0 + canoe[1]*cf1) * cx * (1 - cx - 2*cy),
                (cabins[0]*cf0 + cabins[1]*cf1) * cy * (1 - cy - 2*cx - 2*cz),
                cz * (1 - cz - 2*cy),
                cf0 * (1 - cf0) * (cf0 - 0.5 + 0.05 * (cx*canoe[0] + cy*cabins[0])),
                cf1 * (1 - cf1) * (cf1 - 0.5 + 0.05 * (cx*canoe[1] + cy*cabins[1]))])

# analysis of fixed points
# Agr. Attr. is a fp
sys_c.subs({cx:1, cy:0, cz:1, cf0:1, cf1:1})
sys_s.subs({sx:1, sy:0, sz:1, sf0:1, sf1:1})
sys_alt.subs({cx:1, cy:0, cz:1, cf0:1, cf1:1})

jac_c = sys_c.jacobian([cx, cy, cz, cf0, cf1])
eigs_c = jac_c.subs({cx:1, cy:0, cz:1, cf0:1, cf1:1}).eigenvects()
jac_s = sys_s.jacobian([sx, sy, sz, sf0, sf1])
eigs_s = jac_s.subs({sx:1, sy:0, sz:1, sf0:1, sf1:1}).eigenvects()
jac_alt = sys_alt.jacobian([cx, cy, cz, cf0, cf1])
eigs_alt = jac_alt.subs({cx:1, cy:0, cz:1, cf0:1, cf1:1}).eigenvects()

# Result: 'sailboats' has more negative avg. eigenvals at relevant fp.
# This means that it can more quickly approach that fp, leaving less time
# for noise to push it around
# This is due to a difference in how well N2 overlaps with feature vector of
# NPmod attachment site on N1: it's higher for 'sailboats'
# Putting a 1 for the feature match between n2 and n1:npmod gives identical
# eigenvecs and -vals.

