# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:37:15 2017

@author: garrettsmith

Yet to implement:
    1. Feature passing: dependent representation in a treelet should influence
        the number feature in the same treelet's head bank & vice versa
        (incorp. into W_rec).
    2. Figure out how to handle input from other treelets. Simply adding it in
        as another term won't cut it if we want the activations bounded in
        [0, 1].
"""

import numpy as np
import matplotlib.pyplot as plt

class Treelet(object):
    def __init__(self, nlex, nheadmorph, ndependents, ndepmorph, dim_names):
        self.nlex = nlex
        self.ndependents = ndependents# + ndepmorph
        self.nheadmorph = nheadmorph
        self.ndepmorph = ndepmorph
        self.nfeatures = nlex + nheadmorph + ndependents + ndepmorph
        self.idx = np.arange(self.nfeatures, dtype = 'int')
        self.idx_lex = self.idx[0:self.nlex]
        self.idx_headmorph = self.idx[self.nlex:self.nlex + self.nheadmorph]
        self.idx_dependent = self.idx[self.nlex + self.nheadmorph:self.nlex 
                                      + self.nheadmorph + self.ndependents]
        self.idx_depmorph = np.arange(-2, 0)
        self.idx_head = np.append(self.idx_lex, self.idx_headmorph)
        self.idx_wholedep = np.append(self.idx_dependent, self.idx_depmorph)
        self.state_hist = None
        self.dim_names = dim_names
        self.W_rec = np.zeros((self.nfeatures, self.nfeatures))
    
    def set_recurrent_weights(self):
        """Set recurrent weights with inhibitory connections within banks of
        units. Does not yet set weights between feature banks!"""
        W = np.zeros(self.W_rec.shape)
        W[np.ix_(self.idx_lex, self.idx_lex)] = 2 * np.ones((self.nlex, self.nlex))
        W[np.ix_(self.idx_headmorph, self.idx_headmorph)] = 2 * np.ones((self.nheadmorph, self.nheadmorph))
        if self.ndependents is not 0:
            W[np.ix_(self.idx_dependent, self.idx_dependent)] = 2 * np.ones((self.ndependents, self.ndependents))
            W[np.ix_(self.idx_depmorph, self.idx_depmorph)] = 2 * np.ones((self.ndepmorph, self.ndepmorph))
        np.fill_diagonal(W, 1)
        self.W_rec = W
        
    def random_initial_state(self, noise_mag):
        noisy_init = np.random.uniform(0, noise_mag, self.nfeatures)
        self.set_initial_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeatures, 'Wrong length initial state'
        assert self.state_hist is not None, 'state_hist not initialized'
        self.state_hist[0,] = vec
    
    def print_state(self, t = -1):
        for n in range(len(self.dim_names)):
            print('{}:\t{}'.format(self.dim_names[n], self.state_hist[t,n]))
        print('\n')
        
    def plot_state_hist(self):
        for dim in range(len(self.dim_names)):
            plt.plot(self.state_hist[:,dim], label = self.dim_names[dim])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.ylim(-0.01, 1.01)
        plt.legend()
        plt.title('State over time')
        plt.show()
        
# Trying a single treelet
tstep = 0.01
tvec = np.arange(0.0, 20.0, tstep)

Det = Treelet(2, 2, 0, 0, ['a', 'these', 'det_sg', 'det_pl'])
Det.set_recurrent_weights()
Det.state_hist = np.zeros((len(tvec), Det.nfeatures))
Det.set_initial_state(np.array([0.1, 0.26, 0.11, 0.25]))
#Det.random_initial_state(0.1)

Noun = Treelet(2, 2, 2, 2, ['dog', 'cat', 'n_sg', 'n_pl', 'a', 'these', 'det_sg', 'det_pl'])
Noun.set_recurrent_weights()
Noun.state_hist = np.zeros((len(tvec), Noun.nfeatures))
Noun.random_initial_state(0.1)

link_dn = np.zeros(len(tvec))

for t in range(1, len(tvec)):
    # LV dyn: dx/dt = x * (1 - W_rec @ x)
    link_dn[t] = (Det.state_hist[t-1, Det.idx_head]
    @ Noun.state_hist[t-1, Noun.idx_wholedep]) / (Noun.ndependents + Noun.ndepmorph)
    
#    Det.state_hist[t,] = Det.state_hist[t-1,] + tstep * (Det.state_hist[t-1,]
#    * (1 - Det.W_rec @ Det.state_hist[t-1,]))
    Det.state_hist[t,] = Det.state_hist[t-1,] + tstep * (Det.state_hist[t-1,]
    * (1 - Det.W_rec @ Det.state_hist[t-1,]))
    
    input_from_det = np.ones(Noun.nfeatures)
    input_from_det[Noun.idx_wholedep] = link_dn[t] * Det.state_hist[t-1,Det.idx_head]
    Noun.state_hist[t,] = Noun.state_hist[t-1,] + tstep * (Noun.state_hist[t-1,] 
                   * (1 - Noun.W_rec @ Noun.state_hist[t-1,]))
    
#    if t == 250:
#        Noun.state_hist[t,Noun.idx_head] = np.array([1, 0, 0, 1])
    
Det.plot_state_hist()
Noun.plot_state_hist()
plt.plot(link_dn)
Det.print_state()
Noun.print_state()
