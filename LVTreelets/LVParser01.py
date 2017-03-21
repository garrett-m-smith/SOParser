# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:37:15 2017

@author: garrettsmith

Yet to implement:
    1. Feature passing: dependent representation in a treelet should influence
        the number feature in the same treelet's head bank & vice versa. Need 
        bidirectional influence between dep. and head.
    2. Add PP modifier
    3. Add link dynamics
        
Multiplying both the growth rate and the term current state by input seems to
work, but growth rate only doesn't (Afraimovich et al., 2004; Muezzinoglu et 
al., 2010; Rabinovich et al., 2001). Unless we're willing to let the maximal
activation be less than one, in which case, it's fine to just use the growth
rate alone as the input.

Something to keep in mind: Fukai & Tanaka (1997) show that there is a lower
bound on activations that can make it to winner status; in other words, if a 
feature starts out below that threshold and can't get abov it, it's doomed to 
fall to 0. Pay attn. to for intro'ing phon. form.

One funny thing: A plural deteriminer can eventually force both the noun and 
the verb to become plural, even if they are both introduced as singular...
"""

import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_prod = v1 @ v2
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_prod / denom

class Treelet(object):
#    def __init__(self, nlex, nheadmorph, ndependents, ndepmorph, dim_names,
#                 name, all_words):
    def __init__(self, all_words, all_categories, head_words, name, dim_names):
        assert type(all_words) is list, "Wrong type for all_words."
        assert type(head_words) is list, "Wrong type for head_words."
        self.name = name
        self.nheadlex = len(head_words)
        self.ncat = len(all_categories)
        self.nmorph = 2
        self.nhead = self.nheadlex + self.ncat + self.nmorph
        self.ndeplex = len(all_words)
        self.nfeat = (self.nheadlex + self.ndeplex + 2 * self.ncat + 2 
                      * self.nmorph)
        self.idx = {'head_lex': np.arange(0, self.nheadlex),
                    'head_cat': np.arange(self.nheadlex, self.nheadlex 
                                          + self.ncat),
                    'head_morph': np.arange(self.nheadlex + self.ncat,
                                            self.nheadlex + self.ncat 
                                            + self.nmorph),
                    'head': np.arange(0, self.nhead),
                    'dep_lex': np.arange(self.nhead, self.nhead
                                         + self.ndeplex),
                    'dep_cat': np.arange(self.nhead + self.ndeplex, self.nhead
                                         + self.ndeplex + self.ncat),
                    'dep_morph': np.arange(self.nhead + self.ndeplex 
                                           + self.ncat, self.nhead 
                                           + self.ndeplex + self.ncat 
                                           + self.nmorph),
                    'dep': np.arange(self.nhead, self.nhead 
                                           + self.ndeplex + self.ncat 
                                           + self.nmorph)}
        

#        self.nfeatures = nlex + nheadmorph + ndependents + ndepmorph
#        self.idx = np.arange(self.nfeatures, dtype = 'int')
#        self.idx_lex = self.idx[0:self.nlex]
#        self.idx_headmorph = self.idx[self.nlex:self.nlex + self.nheadmorph]
#        self.idx_dependent = self.idx[self.nlex + self.nheadmorph:self.nlex 
#                                      + self.nheadmorph + self.ndependents]
#        self.idx_depmorph = np.arange(-2, 0)
#        self.idx_head = np.append(self.idx_lex, self.idx_headmorph)
#        self.idx_wholedep = np.append(self.idx_dependent, self.idx_depmorph)
        self.state_hist = None
        assert len(dim_names) == self.nfeat, 'Wrong number of dim_names'
        self.dim_names = dim_names
        self.W_rec = np.zeros((self.nfeat, self.nfeat))
    
    def set_recurrent_weights(self):
        """Set recurrent weights with inhibitory connections within banks of
        units."""
        W = np.zeros(self.W_rec.shape)
        k = 1.5
#        W[np.ix_(self.idx_lex, self.idx_lex)] = k * np.ones((self.nlex, self.nlex))
        W[np.ix_(self.idx['head_lex'], self.idx['head_lex'])] = (k 
         * np.ones((self.nheadlex, self.nheadlex)))
#        W[np.ix_(self.idx_headmorph, self.idx_headmorph)] = k * np.ones((self.nheadmorph, self.nheadmorph))
        W[np.ix_(self.idx['head_cat'], self.idx['head_cat'])] = (k
         * np.ones((self.ncat, self.ncat)))        
        W[np.ix_(self.idx['head_morph'], self.idx['head_morph'])] = (k 
         * np.ones((self.nmorph, self.nmorph)))
#        if self.ndependents is not 0:
        W[np.ix_(self.idx['dep_lex'], self.idx['dep_lex'])] = (k 
         * np.ones((self.ndeplex, self.ndeplex)))
        W[np.ix_(self.idx['dep_cat'], self.idx['dep_cat'])] = (k 
         * np.ones((self.ncat, self.ncat)))
        W[np.ix_(self.idx['dep_morph'], self.idx['dep_morph'])] = (k 
         * np.ones((self.nmorph, self.nmorph)))
        np.fill_diagonal(W, 1)
        self.W_rec = W
        
    def random_initial_state(self, noise_mag):
        """Sets a random initial state drawn from a uniform distribution
        between 0 and noise_mag. Initial conditions need to be somewhat higher
        (here, between 0.1 and 0.1 + noise_mag) in order to make it possible
        for any of the states to win (Fukai & Tanaka, 1997)."""
        noisy_init = np.random.uniform(0.1, 0.1 + noise_mag, self.nfeat)
        self.set_initial_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeat, 'Wrong length initial state'
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
        plt.title('{} state over time'.format(self.name))
        plt.show()
        
#class Lexicon(object):
#    def __init__(self):
#        self.words = []
#        self.categories = []
#        self.lexicon = {}
#    
#    def add_treelet(self, treelet):
#        self.lexicon.update({treelet.name: treelet})
#        for word in treelet
        

# Trying it:
tstep = 0.01
tvec = np.arange(0.0, 70.0, tstep)
words = ['a', 'these', 'that', 'dog', 'cat', 'be', 'sing']
categs = ['Det', 'N', 'V']

#Det = Treelet(3, 2, 0, 0, ['a', 'these', 'that', 'det_sg', 'det_pl'], 'Det')
det_lex = ['a', 'these', 'that']
det_dims = ['a', 'these', 'that', 'Det', 'N', 'V', 'head_sg', 'head_pl', 'a',
            'these', 'that', 'dog', 'cat', 'be', 'sing', 'Det', 'N', 'V', 
            'dep_sg', 'dep_pl']
det_patterns = {'a': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0]),
                'these': np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0]),
                'that': np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0])}
Det = Treelet(words, categs, det_lex, 'Det', det_dims)
Det.set_recurrent_weights()
Det.state_hist = np.zeros((len(tvec), Det.nfeat))
Det.set_initial_state(det_patterns['a'])
#Det.random_initial_state(0.1)

noun_lex = ['dog', 'cat']
noun_dims = ['dog', 'cat', 'Det', 'N', 'V', 'head_sg', 'head_pl', 'a',
            'these', 'that', 'dog', 'cat', 'be', 'sing', 'Det', 'N', 'V', 
            'dep_sg', 'dep_pl']
noun_patterns = {'dog': np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0]),
                'cat': np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0])}
Noun = Treelet(words, categs, noun_lex, 'N', noun_dims)
Noun.set_recurrent_weights()
Noun.state_hist = np.zeros((len(tvec), Noun.nfeat))
Noun.random_initial_state(0.1)

verb_lex = ['be', 'sing']
verb_dims = ['be', 'sing', 'Det', 'N', 'V', 'head_sg', 'head_pl', 'a',
            'these', 'that', 'dog', 'cat', 'be', 'sing', 'Det', 'N', 'V', 
            'dep_sg', 'dep_pl']
noun_patterns = {'be': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0]),
                'sing': np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0])}
Verb = Treelet(words, categs, noun_lex, 'N', noun_dims)
Verb.set_recurrent_weights()
Verb.state_hist = np.zeros((len(tvec), Verb.nfeat))
Verb.random_initial_state(0.1)

link_dn = np.zeros(len(tvec))
link_nv = np.zeros(len(tvec))

for t in range(1, len(tvec)):
    # LV dyn: dx/dt = x * (1 - W_rec @ x)
    link_dn[t] = (Det.state_hist[t-1, Det.idx_head] @  
           Noun.state_hist[t-1, Noun.idx_wholedep]) / 2
    link_nv[t] = (Noun.state_hist[t -1, Noun.idx_head] @
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
        Noun.state_hist[t,Noun.idx_head] = np.array([0.1, 0.9, 0.1, 0.9])
    if t == 1500: # sing
        Verb.state_hist[t,Verb.idx_head] = np.array([0.1, 0.9, 0.1, 0.9])
    
Det.plot_state_hist()
Noun.plot_state_hist()
Verb.plot_state_hist()

plt.plot(link_dn, label = 'Det-N link')
plt.plot(link_nv, label = 'N-V link')
plt.legend()
plt.show()

Det.print_state()
Noun.print_state()
Verb.print_state()
