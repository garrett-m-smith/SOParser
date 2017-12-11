# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 09:25:34 2017

@author: garrettsmith

Going for simplest garden path. Modeling competition between links attaching
"baby" to V1 and "baby" to V2 in:
    "While M. V1 baby V2."

Processing:
    1. Intro V1 slightly perturbed from origin (meas. RT)
    2. Once it settles to origin (speed crit.), intro "baby" by bumping the
    system towards the [1, 0] attractor.
    3. Once speed drops again (record time), intro V2 by bumping in the
    towards [0, 1].i
"""

import numpy as np
import matplotlib.pyplot as plt


def l2norm(x):
    return np.sqrt(x @ x)


def rbf(x, c, gamma):
    phi = np.exp(-l2norm(x-c)**2/gamma)
    return phi


def dyn(x, c, gamma):
    """Takes the current position, a list of attractors, and the gamma
    parameters as arguments and updates the state.
    """
    weights = [0.5, 1., 1.]
    dx = np.zeros(len(x))
    for i in range(len(c)):
        if i == 0:
            dx += 0.5 * (-2./gamma) * (x - c[i]) * rbf(x, c[i], gamma)
        else:
            dx += (-2./gamma) * (x - c[i]) * rbf(x, c[i], gamma)
    return dx


def energy(x, c, gamma):
    e = 0
    for i in range(len(c)):
        e += rbf(x, c[i], gamma)
    return e


def speed(x, c, gamma):
    vel = dyn(x, c, gamma)
    return l2norm(vel)


ndim = 2
tlen = 500
orig = np.array([0., 0])
nV1 = np.array([1., 0])
nV2 = np.array([0., 1])
no_comp = np.array([1., 1.])
gp = [orig, nV1, nV2]
unambig = [orig, nV1, nV2, no_comp]
# Feature matches/size of basins
gamma = 0.25
# Trying out init. cond. a little away from attr. at origin
tau = 0.01
nreps = 1000
isi = 100  # works reasonably well for a range of values
rts = np.zeros(3)
parses = np.zeros(len(gp))

# GP
for rep in range(nreps):
    x = np.zeros((tlen, ndim))
    x[0, ] = [0.25]*ndim
    noise = np.sqrt(tau) * np.random.normal(0, 0.1, (tlen, ndim))
    e = np.zeros(tlen)
    speeds = np.zeros(tlen)
    next_word = 0
    t = 0
    while t < tlen:
        t += 1
        x[t, ] = x[t-1, ]+(tau * dyn(x[t-1, ], gp, gamma) + noise[t-1, ])
        e[t-1, ] = energy(x[t-1, ], gp, gamma)
        speeds[t-1, ] = speed(x[t-1, ], gp, gamma)
        # Once speed drops low enough, bump and get next word
        if speeds[t-1, ] < 0.2 and next_word == 0:
            x[t, ] += nV1# - x[t, ]
            rts[0] += t
            next_word = 1
        elif speeds[t-1, ] < 0.2 and next_word == 1 and t > isi:
            rts[1] += t
            x[t, ] += nV2# - x[t, ]
            next_word = 2
        elif speeds[t-1, ] < 0.2 and next_word == 2 and t > 2*isi:
            rts[2] += t
            last = np.round(x[t, ])
            break
    if (last == orig).all():
        parses[0] += 1
    elif (last == nV1).all():
        parses[1] += 1
    elif (last == nV2).all():
        parses[2] += 1


print('Garden Path:\nRTs:\n\tV1: {}\n\tN1: {}\n\tV2: {}\n'.format(*(rts/nreps)))
print('Parses:\n\tOrigin: {}\n\tnV1: {}\n\tnV2: {}\n'.format(*parses))
if nreps == 1:
    plt.plot(x)
    plt.ylim(-0.1, 1.1)
    plt.plot(np.trim_zeros(e)/np.max(np.trim_zeros(e)), alpha=0.5)
    plt.plot(np.trim_zeros(speeds) / np.max(np.trim_zeros(speeds)), alpha=0.5)
    plt.legend(['x1', 'x2', 'Energy', 'Speed'])
    plt.show()

rts = np.zeros(4)
parses = np.zeros(4)

# Unambig
for rep in range(nreps):
    x = np.zeros((tlen, ndim))
    x[0, ] = [0.25]*ndim
    noise = np.sqrt(tau) * np.random.normal(0, 0.1, (tlen, ndim))
    e = np.zeros(tlen)
    speeds = np.zeros(tlen)
    next_word = 0
    t = 0
    while t < tlen:
        t += 1
        x[t, ] = x[t-1, ]+(tau * dyn(x[t-1, ], unambig, gamma) + noise[t-1, ])
        e[t-1, ] = energy(x[t-1, ], unambig, gamma)
        speeds[t-1, ] = speed(x[t-1, ], unambig, gamma)
        # Once speed drops low enough, bump and get next word
        # Getting N1
        if speeds[t-1, ] < 0.2 and next_word == 0:
            x[t, ] += nV1# - x[t, ]
            rts[0] += t
            next_word = 1
        # Getting N2
        elif speeds[t-1, ] < 0.2 and next_word == 1 and t > isi:
            rts[1] += t
            x[t, ] += nV2# - x[t, ]
            next_word = 2
        # Getting V2
        elif speeds[t-1, ] < 0.2 and next_word == 2 and t > 2*isi:
            rts[2] += t
            x[t, ] += np.random.normal(0, 0.1, ndim)
            next_word = 3
        # Ending
        elif speeds[t-1, ] < 0.2 and next_word == 3 and t > 3*isi:
            rts[3] += t
            last = np.round(x[t, ])
            break
    if (last == orig).all():
        parses[0] += 1
    elif (last == nV1).all():
        parses[1] += 1
    elif (last == nV2).all():
        parses[2] += 1
    elif (last == no_comp).all():
        parses[3] += 1


print('Unambiguous:\nRTs:\n\tV1: {}\n\tN1: {}\n\tN2: {}\n\tV2: {}'.format(*(rts/nreps)))
print('Parses:\n\tOrigin: {}\n\tnV1: {}\n\tnV2: {}\n\tCorrect: {}'.format(*parses))
