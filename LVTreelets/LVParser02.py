# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:37:15 2017

@author: garrettsmith

Things to remember:
    Head treelets don't influence the head bank of their dependents, just the
agr bank.
    No interweights right now, just reliant on phonological input.

25.03.: all of the links are there. Need to implement way of saving all link
strengths and then doing the link competition.

Next, link competition and extend to PP modifiers.
"""

import numpy as np
import matplotlib.pyplot as plt


class Treelet(object):
#    def __init__(self, nlex, nheadmorph, ndependents, ndepmorph, dim_names):
    def __init__(self, lexicon, number, pos, name):
        self.name = name
        # Add extra lex. for "NULL"; x2 b/c dep nodes
        self.nfeat = 2 * (1 + len(lexicon) + len(pos)) + len(number)
        self.nwords = len(lexicon) + 1
        self.npos = len(pos)
        self.nagr = len(number)
        self.nhead = 1 + len(lexicon) + len(pos)
        self.idx = {'head': np.arange(0, self.nhead),
                    'head_lex': np.arange(0, self.nwords),
                    'head_pos': np.arange(self.nwords, self.nwords 
                                          + self.npos),
                    'dep': np.arange(self.nhead, 2 * self.nhead),
                    'dep_lex': np.arange(self.nhead, self.nhead + self.nwords),
                    'dep_pos': np.arange(self.nhead + self.nwords, self.nhead 
                                         + self.nwords + self.npos),
                    'agr': np.arange(2 * self.nhead, 2 * self.nhead 
                                     + len(number)),
                    'head_agr': np.append(np.arange(0, self.nhead), [-2, -1]),
                    'dep_agr': np.append(np.arange(self.nhead, 2 
                                                   * self.nhead), [-2, -1])}
        self.state_hist = None
        self.W_rec = np.zeros((self.nfeat, self.nfeat))
        self.dim_names = (['null'] + lexicon + pos 
                          + ['dep_' + x for x in ['null'] + lexicon] 
                          + ['dep_' + x for x in pos] + number)
    
    def set_recurrent_weights(self):
        """Set recurrent weights with inhibitory connections within banks of
        units."""
        W = np.zeros(self.W_rec.shape)
        k = 2
        W[np.ix_(self.idx['head_lex'], self.idx['head_lex'])] = (k 
         * np.ones((self.nwords, self.nwords)))
        W[np.ix_(self.idx['head_pos'], self.idx['head_pos'])] = (k
         * np.ones((self.npos, self.npos)))
        W[np.ix_(self.idx['dep_lex'], self.idx['dep_lex'])] = (k
         * np.ones((self.nwords, self.nwords)))
        W[np.ix_(self.idx['dep_pos'], self.idx['dep_pos'])] = (k
         * np.ones((self.npos, self.npos)))
        W[np.ix_(self.idx['agr'], self.idx['agr'])] = (k
         * np.ones((self.nagr, self.nagr)))
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
            xpos = dim * (self.state_hist[:,dim].size / len(self.dim_names))
            ypos = self.state_hist[xpos, dim]
            plt.text(xpos, ypos, self.dim_names[dim])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.ylim(-0.01, 1.01)
#        plt.legend(loc = 'center right')
        plt.legend(bbox_to_anchor = (1, 1.03))
        plt.title('{} state over time'.format(self.name))
        plt.show()
    
    def update_state(self, ipt, t, tstep):
        x = self.state_hist[t-1,]
        W = self.W_rec
        self.state_hist[t,] = x + tstep * (x * (ipt - W @ (ipt * x)))
        

# Trying a single treelet
tstep = 0.01
tvec = np.arange(0.0, 100.0, tstep)

lexicon = ['a', 'these', 'that', 'dog', 'cat', 'be', 'sing']
agr = ['sg', 'pl']
pos = ['null', 'Det', 'N', 'V']

lex_rep = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # a
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], # these
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], #that
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], # dog
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], # cat
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # be
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], # sing
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]) # null

Det = Treelet(lexicon, agr, pos, 'Det')
Det.set_recurrent_weights()
Det.state_hist = np.zeros((len(tvec), Det.nfeat))
det_init = np.zeros(Det.nfeat) + np.random.uniform(0.1, 0.2, Det.nfeat)
det_init[Det.idx['head']] = lex_rep[1]
det_init[np.ix_(Det.idx['agr'])] = np.array([0.05, 0.9])
det_init[np.ix_(Det.idx['dep'])] = lex_rep[-1]
Det.set_initial_state(det_init)
#Det.random_initial_state(0.1)

Noun = Treelet(lexicon, agr, pos, 'Noun')
Noun.set_recurrent_weights()
Noun.state_hist = np.zeros((len(tvec), Noun.nfeat))
Noun.random_initial_state(0.1)
# Nouns expect a determiner as a dependent
Noun.state_hist[0, np.ix_(Noun.idx['dep_pos'])] = np.array([0.1, 0.5, 0.1, 0.1])

Verb = Treelet(lexicon, agr, pos, 'Verb')
Verb.set_recurrent_weights()
Verb.state_hist = np.zeros((len(tvec), Verb.nfeat))
Verb.random_initial_state(0.1)
# Verbs expect a Noun as a dependent
Verb.state_hist[0, np.ix_(Verb.idx['dep_pos'])] = np.array([0.1, 0.1, 0.5, 0.1])

#link_dn = np.zeros(len(tvec))
#link_nv = np.zeros(len(tvec))

all_words = [Det, Noun, Verb]

for t in range(1, len(tvec)):
    for word in all_words:
        others = [x for x in all_words if x is not word]
        to_word = np.ones(word.nfeat)
        to_word[word.idx['dep_agr']] = 0
        for other in others:
            to_word[word.idx['dep']] += ((word.state_hist[t-1,word.idx['dep']]
            @ other.state_hist[t-1, other.idx['head']] 
            / (word.nhead + word.nagr))
            * other.state_hist[t-1, other.idx['head']])
        to_word = to_word / len(others)
        word.update_state(to_word, t, tstep)
    
#    link_dn[t] = (Det.state_hist[t-1, Det.idx['head_agr']] @  
#           Noun.state_hist[t-1, Noun.idx['dep_agr']]) / (Det.nhead + Det.nagr)
#    link_nv[t] = (Noun.state_hist[t-1, Noun.idx['head_agr']] @
#           Verb.state_hist[t-1, Verb.idx['dep_agr']]) / (Noun.nhead + Noun.nagr)
    
    if tvec[t] == 10: # cat
        Noun.state_hist[t, Noun.idx['head']] = lex_rep[4]
        Noun.state_hist[t, Noun.idx['agr']] = np.array([0, 1])
    if tvec[t] == 20: # sings
        Verb.state_hist[t, Verb.idx['head']] = lex_rep[-2]
        Verb.state_hist[t, Verb.idx['agr']] = np.array([0, 1])
    
Det.plot_state_hist()
Noun.plot_state_hist()
Verb.plot_state_hist()

#plt.plot(link_dn)
#plt.title('Det-N link')
#plt.ylim(-0.01, 0.3)
#plt.show()
#
#plt.plot(link_nv)
#plt.title('N-V link')
#plt.ylim(-0.01, 0.3)
#plt.show()

Det.print_state()
Noun.print_state()
Verb.print_state()
