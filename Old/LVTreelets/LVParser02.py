# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:37:15 2017

@author: garrettsmith

Things to remember:
    No interweights right now, just reliant on phonological input. This might
not be the best thing if we want correct priming...
    Link competition current has the form x * (input - W(ipt * x)).
    Link weight matrix is currently set by hand. Will want to change when
adding more treelets.
    Null mask is implemented, doesn't have the intended effects...
    

Next:
    Done: Start states between 0 and 1
    Half done, has weird effects...: Intro'ing phonological forms should leave
    unaffected features/units unchanged in their activation
    3. Interconnections
    4. Extend to PP modifiers.
    5. Noise
    6. Monte-Carlo mechanism
"""

import numpy as np
import matplotlib.pyplot as plt


class Treelet(object):
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
    
    def shrink_initial_state(self):
        vec = self.state_hist[0,] + np.random.uniform(0.05, 0.15, self.nfeat)
        vec[vec >= 1] = 0.9
        self.state_hist[0,] = vec
    
    def print_state(self, t = -1):
        longest = np.max([len(x) for x in self.dim_names])
        for n in range(len(self.dim_names)):
            print('{:{width}}: {}'.format(self.dim_names[n],
                  np.round(self.state_hist[t,n], 5), width = longest))
        print('\n')
    
    def intro_phon(self, vec, idx, t):
        """Expects a bit vector with the phonological form activated, i.e.,
        set to 1. Sets state_hist at time point t to 0.9 times those values."""
        vec2 = np.clip(self.state_hist[t, self.idx[idx]], 0, 0.8)
#        vec2[vec == 1] = 0.9
        vec2[vec == 1] = 1.
        self.state_hist[t, self.idx[idx]] = vec2
        
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
        

def plot_links(links, all_treelets):
    """Gotta fix!"""
    for treelet in range(len(all_treelets)):
        curr_word = all_treelets[treelet]
        others = [x for x in all_treelets if x is not curr_word]
        for other in range(len(others)):
            curr_other = others[other]
            linked_treelets = (curr_word.name + '-' 
                               + curr_other.name)
            plt.plot(links[treelet, other, :], label = linked_treelets)
        plt.xlabel('Time')
        plt.ylabel('Link strength')
        plt.legend(loc = 'center right')
        plt.title("Links to {}'s dependent attachment site".format(all_treelets[treelet].name))
        plt.ylim(-0.01, 1.01)
        plt.show()

def update_links(links, overlap, Wlinks, t, tstep):
    x = links[:, :, t-1].flatten()
    ipt = overlap.flatten()
    xt1 = x + tstep * (x * (ipt - Wlinks @ (ipt * x)))
#    xt1 = x + tstep * (x * (ipt - Wlinks @ x))
    links[:, :, t] = xt1.reshape(links.shape[0], links.shape[1])


# Here we go:
tstep = 0.01
tvec = np.arange(0.0, 150.0, tstep)

lexicon = ['a', 'these', 'that', 'dog', 'cat', 'be', 'sing']
agr = ['sg', 'pl']
pos = ['null', 'Det', 'N', 'V']

# Lexical representations: first lexical units, then POS units
lex_rep = np.array([[0., 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # a
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
det_init[np.ix_(Det.idx['agr'])] = np.array([0, 1.])
det_init[np.ix_(Det.idx['dep'])] = lex_rep[-1]
Det.set_initial_state(det_init)
#Det.shrink_initial_state()
#Det.random_initial_state(0.1)

Noun = Treelet(lexicon, agr, pos, 'Noun')
Noun.set_recurrent_weights()
Noun.state_hist = np.zeros((len(tvec), Noun.nfeat))
Noun.random_initial_state(0.1)
# Nouns expect a determiner as a dependent
Noun.state_hist[0, np.ix_(Noun.idx['dep_pos'])] = np.array([0, 1., 0, 0])
#Noun.shrink_initial_state()

Verb = Treelet(lexicon, agr, pos, 'Verb')
Verb.set_recurrent_weights()
Verb.state_hist = np.zeros((len(tvec), Verb.nfeat))
Verb.random_initial_state(0.1)
# Verbs expect a Noun as a dependent
Verb.state_hist[0, np.ix_(Verb.idx['dep_pos'])] = np.array([0, 0, 1., 0])
#Verb.shrink_initial_state()


all_words = [Det, Noun, Verb]
overlap = np.zeros((len(all_words), len(all_words) - 1))
links = np.zeros((len(all_words), len(all_words) - 1, len(tvec)))
#links[:,:,0] = np.random.uniform(0.1, 0.2, links[:,:,0].shape)

# Links from same head should compete, links to same dependent should compete,
# and one only treelet should dominate.
k = 2
W_links = np.array([[1, k, k, 0, 0, k],
                    [k, 1, 0, k, k, 0],
                    [k, 0, 1, k, k, 0],
                    [0, k, k, 1, 0, k],
                    [0, k, k, 0, 1, k],
                    [k, 0, 0, k, k, 1]])

null_filter = np.ones(Det.nhead + Det.nagr)
#null_filter[0] = 0
#null_filter[Det.idx['head_pos'][0]] = 0

# Setting initial link conditions to the first overlap * 0.5
for word in range(len(all_words)):
    curr_word = all_words[word]
    others = [x for x in all_words if x is not curr_word]
    for other in range(len(others)):
        curr_other = others[other]
        links[word, other, 0] = ((null_filter 
            * curr_word.state_hist[0, curr_word.idx['dep_agr']])
            @ (null_filter * curr_other.state_hist[0, curr_other.idx['head_agr']])
            / (curr_word.nhead + curr_word.nagr)) * 0.5

for t in range(1, len(tvec)):
    for word in range(len(all_words)):
        curr_word = all_words[word]
        others = [x for x in all_words if x is not curr_word]
        to_word = np.ones(curr_word.nfeat)
        to_word[curr_word.idx['dep_agr']] = 0
        for other in range(len(others)):
            curr_other = others[other]
            # Calculate overlap
            overlap[word, other] = ((null_filter * curr_word.state_hist[t-1,
                   curr_word.idx['dep_agr']])
            @ (null_filter * curr_other.state_hist[t-1, curr_other.idx['head_agr']]) 
            / (curr_word.nhead + curr_word.nagr))
        
        # Doing link competition
        update_links(links, overlap, W_links, t, tstep)
        
        # Getting link-weighted input from other treelets
        for other in range(len(others)):
            curr_other = others[other]
            # Add in link-strength-weighted contrib from sending treelet
            to_word[curr_word.idx['dep_agr']] += (links[word, other, t]
            * (null_filter * curr_other.state_hist[t-1, curr_other.idx['head_agr']]))
            to_word[curr_word.idx['head_agr']] += (links[word, other, t]
            * (null_filter * curr_other.state_hist[t-1, curr_other.idx['dep_agr']]))
        
        # Avg.ing/normalizing
        to_word = to_word / (len(all_words) - 1)
        # LV treelet dynamics
        all_words[word].update_state(to_word, t, tstep)
    
    # Inputing phonology
    if tvec[t] == 30:
        Noun.intro_phon(lex_rep[3,], 'head', t)
        Noun.intro_phon(np.array([0, 1.]), 'agr', t)
    if tvec[t] == 60:
        Verb.intro_phon(lex_rep[-2], 'head', t)
        Verb.intro_phon(np.array([0., 1.]), 'agr', t)
    
# Checking the results:
Det.plot_state_hist()
Noun.plot_state_hist()
Verb.plot_state_hist()

plot_links(links, all_words)

Det.print_state()
Noun.print_state()
Verb.print_state()
