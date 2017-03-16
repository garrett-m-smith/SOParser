# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:37:15 2017

@author: garrettsmith

Yet to implement:
    1. Feature passing: dependent representation in a treelet should influence
        the number feature in the same treelet's head bank & vice versa
        (incorp. into W_rec).
        
Multiplying both the growth rate and the term current state by input seems to
work, but growth rate only doesn't (Afraimovich et al., 2004; Muezzinoglu et 
al., 2010; Rabinovich et al., 2001)

Something to keep in mind: Fukai & Tanaka (1997) show that there is a lower
bound on activations that can make it to winner status; in other words, if a 
feature starts out below that threshold and can't get abov it, it's doomed to 
fall to 0. Pay attn. to for intro'ing phon. form.
"""

import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_prod = v1 @ v2
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_prod / denom

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
        k = 1.5
        W[np.ix_(self.idx_lex, self.idx_lex)] = k * np.ones((self.nlex, self.nlex))
        W[np.ix_(self.idx_headmorph, self.idx_headmorph)] = k * np.ones((self.nheadmorph, self.nheadmorph))
        if self.ndependents is not 0:
            W[np.ix_(self.idx_dependent, self.idx_dependent)] = k * np.ones((self.ndependents, self.ndependents))
            W[np.ix_(self.idx_depmorph, self.idx_depmorph)] = k * np.ones((self.ndepmorph, self.ndepmorph))
        np.fill_diagonal(W, 1)
        self.W_rec = W
        
    def random_initial_state(self, noise_mag):
        """Sets a random initial state drawn from a uniform distribution
        between 0 and noise_mag. Initial conditions need to be somewhat higher
        (here, between 0.1 and 0.1 + noise_mag) in order to make it possible
        for any of the states to win (Fukai & Tanaka, 1997)."""
        noisy_init = np.random.uniform(0.1, 0.1 + noise_mag, self.nfeatures)
        self.set_initial_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeatures, 'Wrong length initial state'
        assert self.state_hist is not None, 'state_hist not initialized'
        self.state_hist[0,] = vec
    
    def print_state(self, t = -1):
        longest = np.max([len(x) for x in self.dim_names])
        for n in range(len(self.dim_names)):
            print('{:{width}}: {}'.format(self.dim_names[n],
                  np.round(self.state_hist[t,n], 5), width = longest))
        print('\n')
        
    def plot_state_hist(self):
        for dim in range(len(self.dim_names)):
            plt.plot(self.state_hist[:,dim], label = self.dim_names[dim])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.ylim(-0.01, 1.01)
        plt.legend(loc = 'center right')
        plt.title('State over time')
        plt.show()
        
# Trying a single treelet
tstep = 0.01
tvec = np.arange(0.0, 50.0, tstep)

Det = Treelet(3, 2, 0, 0, ['a', 'these', 'that', 'det_sg', 'det_pl'])
Det.set_recurrent_weights()
Det.state_hist = np.zeros((len(tvec), Det.nfeatures))
Det.set_initial_state(np.array([0.05, 1, 0.05, 0.05, 1]))
#Det.random_initial_state(0.1)

Noun = Treelet(2, 2, 3, 2, ['dog', 'cat', 'n_sg', 'n_pl', 'a', 'these', 'that', 'det_sg', 'det_pl'])
Noun.set_recurrent_weights()
Noun.state_hist = np.zeros((len(tvec), Noun.nfeatures))
Noun.random_initial_state(0.1)

Verb = Treelet(2, 2, 2, 2, ['be', 'sing', 'v_sg', 'v_pl', 'dog', 'cat', 'subj_sg', 'subj_pl'])
Verb.set_recurrent_weights()
Verb.state_hist = np.zeros((len(tvec), Verb.nfeatures))
Verb.random_initial_state(0.1)

link_dn = np.zeros(len(tvec))
link_nv = np.zeros(len(tvec))

for t in range(1, len(tvec)):
    # LV dyn: dx/dt = x * (1 - W_rec @ x)
    link_dn[t] = (Det.state_hist[t-1, Det.idx_head] @  
           Noun.state_hist[t-1, Noun.idx_wholedep]) / 2
    link_nv[t] = (Noun.state_hist[t-1, Noun.idx_head] @
           Verb.state_hist[t-1, Verb.idx_wholedep]) / 2
#    link_dn[t], link_nv[t] = (1, 1)
    
    input_from_n = np.ones(Det.nfeatures)
    input_from_n[Det.idx_head] = link_dn[t] * Noun.state_hist[t-1, Noun.idx_wholedep]
    Det.state_hist[t,] = Det.state_hist[t-1,] + tstep * (Det.state_hist[t-1,] 
    * (input_from_n - Det.W_rec @ (Det.state_hist[t-1,] * input_from_n)))
    
    input_to_n = np.ones(Noun.nfeatures)
    input_to_n[Noun.idx_wholedep] = link_dn[t] * Det.state_hist[t-1,]
    input_to_n[Noun.idx_head] = link_nv[t] * Verb.state_hist[t-1, Verb.idx_wholedep]
    input_to_n[Noun.idx_headmorph] = (input_to_n[Noun.idx_headmorph]
    + Noun.state_hist[t-1,Noun.idx_depmorph]) / 2
    Noun.state_hist[t,] = Noun.state_hist[t-1,] + tstep * (Noun.state_hist[t-1,] 
    * (input_to_n - Noun.W_rec @ (Noun.state_hist[t-1,] * input_to_n)))
    
    input_to_verb = np.ones(Verb.nfeatures)
    input_to_verb[Verb.idx_wholedep] = link_nv[t] * Noun.state_hist[t-1,Noun.idx_head]
    input_to_verb[Verb.idx_headmorph] = (input_to_verb[Verb.idx_headmorph]
    + Verb.state_hist[t-1,Verb.idx_depmorph]) / 2
    Verb.state_hist[t,] = Verb.state_hist[t-1,] + tstep * (Verb.state_hist[t-1,] 
    * (input_to_verb - Verb.W_rec @ (Verb.state_hist[t-1,] * input_to_verb)))
    
    if t == 500: # cats
        Noun.state_hist[t,Noun.idx_head] = np.array([0.05, 0.95, 0.05, 0.95])
    if t == 1500: # sing
        Verb.state_hist[t,Verb.idx_head] = np.array([0.01, 0.9, 0.01, 0.9])
    
Det.plot_state_hist()
Noun.plot_state_hist()
Verb.plot_state_hist()

plt.plot(link_dn)
plt.title('Det-N link')
plt.ylim(-0.01, 1.01)
plt.show()

plt.plot(link_nv)
plt.title('N-V link')
plt.ylim(-0.01, 1.01)
plt.show()

Det.print_state()
Noun.print_state()
Verb.print_state()
