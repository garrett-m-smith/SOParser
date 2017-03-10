# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:30:12 2017

@author: garrettsmith

General equations:
    da/dt = -a + f(Wa + I)
    I = link_strength * features_to_match

features_to_match is the vector of features from the treelet at the other end
of the link. Thus, feature passing is implemented as a term in the net
input to the unit, along with the recurrent dynamics (recurrent matrix W,
within-treelet activations a) 

I think the treelet dynamics are going too fast, but the parse formation seems
to be getting more reliable, probably due to the corrected interactions between
the treelets. Reliably forms correct parses, but always via garden pathing...

Another problem: it basically always gets garden pathed due to the random
initial conditions... Although it does get to the right parse once
phonological form comes in.

Next Steps:
    0. Implement way of setting recurrent weights
    1. Expand vocab
    2. General link competition
    3. Extend noun treelet with PP modifier (and create PP class)
    
Considerations:
    1. How is it handling less than optimal inputs? What states does it end
       up in?
    2. Relatedly, can I define a harmony/energy function for the whole system?
    3. What initial conditions, time constants, etc., make functional parsing
       possible?

Possible changes:
    1. Switch to sigmoid activation fn. (would necessitate rethinking 
       representation codings and recurrent weight matrices)
    2. Only do similarity calculations on relevant dimensions in a non-kludgy
       way...
"""

import numpy as np
import matplotlib.pyplot as plt


##### Defining utility functions #####
def scalar_proj(vec, pattern_mat):
    return (vec @ pattern_mat) / np.linalg.norm(pattern_mat, axis = 0)

def shepard_similarity(vec, pattern_mat):
    """Calculate the similarity s(x, y) = exp(-|x - y|**2) (Shepard, 1987)"""
#    return np.exp(-np.linalg.norm(pattern_mat - vec, axis = 1)**2)
    return np.exp(-np.linalg.norm(pattern_mat - vec, axis = 1))

def cosine_similarity(vec, pattern_mat):
    """Calculate cosine similarity between a vector and each column in a 
    matrix. Thus, each pattern should be stored as column vectors in the
    matrix."""
    dot_prod = vec @ pattern_mat.T
    denom = np.linalg.norm(vec) * np.linalg.norm(pattern_mat, axis = 1)
    return dot_prod / denom

def plot_trajectories(tvec, similarity, labels=None):
    """Plots similarity trajectories."""
    if labels is not None:
        for i in range(len(labels)):
            plt.plot(tvec, similarity[:, i], label = '{}'.format(labels[i]))
    else:
        plt.plot(tvec, similarity[:, i])
    plt.xlabel('Time')
    plt.ylabel('Similarity')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Similarity')
    plt.show()

class Treelet(object):
    def __init__(self, nlex, ndependents, nlicensors, nmorph):
        self.nlex = nlex
        self.ndependents = ndependents
        self.nlicensors = nlicensors
        self.nmorph = nmorph
        self.nfeatures = nlex + ndependents + nlicensors + nmorph
        self.idx_lex = np.arange(0, nlex)
        self.idx_dependent = np.arange(nlex, nlex + ndependents)
        self.idx_licensor = np.arange(nlex + ndependents, nlex + ndependents + nmorph)
#        self.idx_morph = np.arange(nlex + ndependents + nmorph, self.nfeatures)
        # Kludgy, but...
        self.idx_morph = [-2, -1]
        self.state_hist = None
        
        self.W_rec = np.zeros((self.nfeatures, self.nfeatures))
    
    def set_recurrent_weights(self, pattern_mat):
        """Sets the within-treelet recurrent weights. Assumes pattern_mat has 
        the patterns on each row."""
        W = np.zeros(self.W_rec.shape)
        Wlex = -np.ones((self.nlex, self.nlex))
        np.fill_diagonal(Wlex, 1)
        W[np.ix_(self.idx_lex, self.idx_lex)] = Wlex
        
        if self.ndependents is not 0:
            Wdep = -np.ones((self.ndependents, self.ndependents))
            np.fill_diagonal(Wdep, 1)
            W[np.ix_(self.idx_dependent, self.idx_dependent)] = Wdep
             
        if self.nlicensors is not 0:
            Wlic = -np.ones((self.nlicensors, self.nlicensors))
            np.fill_diagonal(Wlic, 1)
            W[np.ix_(self.idx_licensor, self.idx_licensor)] = Wlic
        
        Wm = -np.ones((self.nmorph, self.nmorph))
        np.fill_diagonal(Wm, 1)
        W[np.ix_(self.idx_morph, self.idx_morph)] = Wm
        self.W_rec = W
        
    def random_initial_state(self, noise_mag):
        noisy_init = np.random.uniform(-noise_mag, noise_mag, self.state.shape)
        self.set_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeatures, 'Wrong length initial state'
        assert self.state_hist is not None, 'state_hist not initialized'
        self.state_hist[0,] = vec

##### Setting up treelets #####
tstep = 0.01
tvec = np.arange(0.0, 30.0, tstep)
ndet = 3
nnoun = 2
nverb = 2

det_patterns = np.array([[1, -1, -1, 0, 0, 1, -1], # a
                         [-1, 1, -1, 0, 0, -1, 1], # these
                         [-1, -1, 1, 0, 0, 1, -1]])#.T # this
noun_patterns = np.array([[1, -1, 0, 0, 0, 0, 0, 1, -1], # dog
                          [1, -1, 0, 0, 0, 0, 0, -1, 1], # dogs
                          [-1, 1, 0, 0, 0, 0, 0, 1, -1], # cat
                          [-1, 1, 0, 0, 0, 0, 0, -1, 1]])#.T # cats
verb_patterns = np.array([[1, -1, 0, 0, 1, -1], # is
                          [1, -1, 0, 0, -1, 1], # are
                          [-1, 1, 0, 0, 1, -1], # sings
                          [-1, 1, 0, 0, -1, 1]])#.T # sing


Det = Treelet(ndet, 0, nnoun, 2)
Det.state_hist = np.zeros((len(tvec), Det.nfeatures))
Det.set_recurrent_weights(det_patterns)
#Det.W_rec = np.array([[1, -1, -1, 0, 0, 1, -1],
#                      [-1, 1, -1, 0, 0, -1, 1],
#                      [-1, -1, 1, 0, 0, 1, -1],
#                      [0, 0, 0, 1, -1, 0, 0],
#                      [0, 0, 0, -1, 1, 0, 0],
#                      [1, -1, 1, 0, 0, 1, -1],
#                      [-1, 1, -1, 0, 0, -1, 1]])
det_init = np.array([-1, 1, -1, 0, 0, -1, 1]) # activating phonology for 'these'
Det.set_initial_state(det_init)

# Noun treelet:
Noun = Treelet(nnoun, ndet, nverb, 2)
Noun.state_hist = np.zeros((len(tvec), Noun.nfeatures))

Noun.set_recurrent_weights(noun_patterns)
#Noun.W_rec = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0],
#                       [-1, 1, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 1, -1, -1, 0, 0, 0, 0],
#                       [0, 0, -1, 1, -1, 0, 0, 0, 0],
#                       [0, 0, -1, -1, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 1, -1, 1, -1],
#                       [0, 0, 0, 0, 0, -1, 1, -1, 1],
#                       [0, 0, 0, 0, 0, 1, -1, 1, -1],
#                       [0, 0, 0, 0, 0, -1, 1, -1, 1]])
Noun.set_initial_state(np.random.uniform(-0.01, 0.01, Noun.nfeatures))

# Verb treelet
Verb = Treelet(nverb, nnoun, 0, 2)
Verb.state_hist = np.zeros((len(tvec), Verb.nfeatures))

Verb.set_recurrent_weights(verb_patterns)
#Verb.W_rec = np.array([[1, -1, 0, 0, 1, -1],
#                       [-1, 1, 0,0, -1, 1],
#                       [0, 0, 1, -1, 0, 0],
#                       [0, 0, -1, 1, 0, 0],
#                       [1, -1, 0, 0, 1, -1],
#                       [-1, 1, 0, 0, -1, 1]])
Verb.set_initial_state(np.random.uniform(-0.01, 0.01, Verb.nfeatures))


##### Running the whole system #####
det_sim = np.zeros((len(tvec), det_patterns.shape[0]))
noun_sim = np.zeros((len(tvec), noun_patterns.shape[0]))
verb_sim = np.zeros((len(tvec), verb_patterns.shape[0]))

link_dn = np.zeros(len(tvec))
link_nv = np.zeros(len(tvec))

for t in range(1, len(tvec)):    
    # Still kludge-y

    # Determiner treelet:
    input_from_n = np.zeros(Det.nfeatures)
    input_from_n[Det.idx_licensor] = Noun.state_hist[t-1, Noun.idx_lex]
    input_from_n[Det.idx_lex] = Noun.state_hist[t-1, Noun.idx_dependent]
    input_from_n[Det.idx_morph] = Noun.state_hist[t-1, Noun.idx_morph]
    
    # Link strength between determiner and noun
    link_dn[t] = (Det.state_hist[t-1,] @ input_from_n) / Det.nfeatures

    # Determiner dynamics:
    Det.state_hist[t,] = Det.state_hist[t-1,] + tstep * (-Det.state_hist[t-1,]
        + np.tanh(Det.W_rec @ Det.state_hist[t-1,] + link_dn[t] * input_from_n))

    # Calculating the similarity:
    det_sim[t,:] = shepard_similarity(Det.state_hist[t,], det_patterns)
#    det_sim[t,:] = cosine_similarity(Det.state_hist[t,], det_patterns)

    # Noun treelet
    input_from_det = np.zeros(Noun.nfeatures)
    input_from_det[Noun.idx_lex] = Det.state_hist[t, Det.idx_licensor]
    input_from_det[Noun.idx_dependent] = Det.state_hist[t, Det.idx_lex]
    input_from_det[Noun.idx_morph] = Det.state_hist[t, Det.idx_morph]

    input_from_verb = np.zeros(Noun.nfeatures)
    input_from_verb[Noun.idx_licensor] = Verb.state_hist[t, Verb.idx_lex]
    input_from_verb[Noun.idx_lex] = Verb.state_hist[t, Verb.idx_dependent]
    input_from_verb[Noun.idx_morph] = Verb.state_hist[t, Verb.idx_morph]

    Noun.state_hist[t,] = Noun.state_hist[t-1,] + tstep * (-Noun.state_hist[t-1,]
        + np.tanh(Noun.W_rec @ Noun.state_hist[t-1,] + link_dn[t] * input_from_det
                  + link_nv[t] * input_from_verb))
    
    noun_sim[t,:] = shepard_similarity(Noun.state_hist[t,], noun_patterns)
    
    # Verb treelet
    input_to_verb = np.zeros(Verb.nfeatures)
    input_to_verb[Verb.idx_lex] = Noun.state_hist[t, Noun.idx_licensor]
    input_to_verb[Verb.idx_dependent] = Noun.state_hist[t, Noun.idx_lex]
    link_nv[t] = (Verb.state_hist[t-1,] @ input_to_verb) / Verb.nfeatures

    Verb.state_hist[t,] = Verb.state_hist[t-1,] + tstep * (-Verb.state_hist[t-1,]
        + np.tanh(Verb.W_rec @ Verb.state_hist[t-1,] + link_nv[t] * input_to_verb))
    verb_sim[t,] = shepard_similarity(Verb.state_hist[t,], verb_patterns)

    # Introduce the noun at t = 5
    if tvec[t] == 5:
        Noun.state_hist[t,] += noun_patterns[1,]

    # Introduce verb at t = 10
    if tvec[t] == 10:
        Verb.state_hist[t,] += verb_patterns[1,]


##### Plotting #####
det_labels = ['a', 'these', 'this']
plot_trajectories(tvec, det_sim, det_labels)

noun_labels = ['dog', 'dogs', 'cat', 'cats']
plot_trajectories(tvec, noun_sim, noun_labels)

verb_labels = ['is', 'are', 'sings', 'sing']
plot_trajectories(tvec, verb_sim, verb_labels)

plt.plot(tvec, link_dn)
plt.ylim(-1, 1)
plt.title('Determiner-noun link strength')
plt.show()

plt.plot(tvec, link_nv)
plt.ylim(-1, 1)
plt.title('Noun-verb link strength')
plt.show()
