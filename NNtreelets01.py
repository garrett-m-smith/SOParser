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
the treelets.

Another problem: it basically always gets garden pathed due to the random
initial conditions... Although it does get to the right parse once
phonological form comes in.

Next Steps:
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
    return np.exp(-np.linalg.norm(pattern_mat - vec, axis = 1)**2,)

def cosine_similarity(vec, pattern_mat):
    """Calculate cosine similarity between a vector and each column in a 
    matrix. Thus, each pattern should be stored as column vectors in the
    matrix."""
    dot_prod = vec @ pattern_mat
    denom = np.linalg.norm(vec) * np.linalg.norm(pattern_mat, axis = 0)
    return dot_prod / denom

def plot_trajectories(tvec, similarity, labels=None):
    """Plots similarity trajectories."""
    for i in range(similarity.shape[1]):
        if labels is not None:
            plt.plot(tvec, similarity[:, i], label = '{}'.format(labels[i]))
        else:
            plt.plot(tvec, similarity[:, i])
    plt.xlabel('Time')
    plt.ylabel('Similarity')
    plt.legend()
    plt.title('Similarity')
    plt.show()

class Treelet(object):
    def __init__(self, nlex, ndependents, nlicensors, nmorph):
        self.nlex = nlex
        self.ndependents = ndependents
        self.nlicensors = nlicensors
        self.nmorph = nmorph
        self.state = np.zeros(nlex + ndependents + nlicensors + nmorph)
        self.nfeatures = len(self.state)
        self.idx_lex = np.arange(0, nlex)
        self.idx_dependent = np.arange(nlex, nlex + ndependents)
        self.idx_licensor = np.arange(nlex + ndependents, nlex + ndependents + nmorph)
#        self.idx_morph = np.arange(nlex + ndependents + nmorph, self.nfeatures)
        # Kludgy, but...
        self.idx_morph = [-2, -1]
        self.state_hist = None
        
        self.W_rec = np.zeros((self.nfeatures, self.nfeatures))
        
    def set_state(self, vals):
        assert vals.shape == self.state.shape
        self.state = vals
        return self.state
    
    def set_recurrent_weights(self):
        """Set recurrent weights with inhibitory connections within banks of
        units. Does not yet set weights between feature banks!"""
        #assert W.shape == self.W_rec.shape
        #self.W_rec = W
        #return self.W_rec
        W = np.zeros(self.W_rec.shape)
        W[np.ix_(self.idx_lex, self.idx_lex)] = -np.ones((self.nlex, self.nlex))
        W[np.ix_(self.idx_dependent, self.idx_dependent)] = -np.ones((self.ndependents, self.ndependents))
        W[np.ix_(self.idx_licensor, self.idx_licensor)] = -np.ones((self.nlicensors, self.nlicensors))
        W[np.ix_(self.idx_morph, self.idx_morph)] = -np.ones((self.nmorph, self.nmorph))
        np.fill_diagonal(W, 1)
        self.W_rec = W
        return self.W_rec
        
    def random_initial_state(self, noise_mag):
        noisy_init = np.random.uniform(-noise_mag, noise_mag, self.state.shape)
        self.set_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeatures, 'Wrong length initial state'
        assert self.state_hist is not None, 'state_hist not initialized'
        self.state_hist[0,] = vec

##### Setting up treelets #####
tstep = 0.01
tvec = np.arange(0.0, 100.0, tstep)
Det = Treelet(3, 0, 2, 2)
Det.state_hist = np.zeros((len(tvec), Det.nfeatures))
det_patterns = np.array([[1, -1, -1, 0, 0, 1, -1], # a
                         [-1, 1, -1, 0, 0, -1, 1], # these
                         [-1, -1, 1, 0, 0, 1, -1]]).T # this
Det.W_rec = np.array([[1, -1, -1, 0, 0, 1, -1],
                      [-1, 1, -1, 0, 0, -1, 1],
                      [-1, -1, 1, 0, 0, 1, -1],
                      [0, 0, 0, 1, -1, 0, 0],
                      [0, 0, 0, -1, 1, 0, 0],
                      [1, -1, 1, 0, 0, 1, -1],
                      [-1, 1, -1, 0, 0, -1, 1]])
det_init = np.array([-1, 1, -1, 0, 0, -1, 1]) # activating phonology for 'these'
Det.set_initial_state(det_init)

# Noun treelet:
Noun = Treelet(2, 3, 2, 2)
Noun.state_hist = np.zeros((len(tvec), Noun.nfeatures))
noun_patterns = np.array([[1, -1, 0, 0, 0, 0, 0, 1, -1], # dog
                          [1, -1, 0, 0, 0, 0, 0, -1, 1], # dogs
                          [-1, 1, 0, 0, 0, 0, 0, 1, -1], # cat
                          [-1, 1, 0, 0, 0, 0, 0, -1, 1]]).T # cats
Noun.W_rec = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0],
                       [-1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, -1, -1, 0, 0, 0, 0],
                       [0, 0, -1, 1, -1, 0, 0, 0, 0],
                       [0, 0, -1, -1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, -1, 1, -1],
                       [0, 0, 0, 0, 0, -1, 1, -1, 1],
                       [0, 0, 0, 0, 0, 1, -1, 1, -1],
                       [0, 0, 0, 0, 0, -1, 1, -1, 1]])
Noun.set_initial_state(np.random.uniform(-0.01, 0.01, Noun.nfeatures))

# Verb treelet
Verb = Treelet(2, 2, 0, 2)
Verb.state_hist = np.zeros((len(tvec), Verb.nfeatures))
verb_patterns = np.array([[1, -1, 0, 0, 1, -1], # is
                          [-1, 1, 0, 0, -1, 1]]).T # are
Verb.W_rec = np.array([[1, -1, 0, 0, 1, -1],
                       [-1, 1, 0,0, -1, 1],
                       [0, 0, 1, -1, 0, 0],
                       [0, 0, -1, 1, 0, 0],
                       [1, -1, 0, 0, 1, -1],
                       [-1, 1, 0, 0, -1, 1]])
Verb.set_initial_state(np.random.uniform(-0.01, 0.01, Verb.nfeatures))
    


##### Running the whole system #####
tstep = 0.01

det_sim = np.zeros((len(tvec), det_patterns.shape[1]))
noun_sim = np.zeros((len(tvec), noun_patterns.shape[1]))
verb_sim = np.zeros((len(tvec), verb_patterns.shape[1]))

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
    det_sim[t,:] = shepard_similarity(Det.state_hist[t,] ,det_patterns.T)

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
    
    noun_sim[t,:] = shepard_similarity(Noun.state_hist[t,], noun_patterns.T)
    
    # Verb treelet
    input_to_verb = np.zeros(Verb.nfeatures)
    input_to_verb[Verb.idx_lex] = Noun.state_hist[t, Noun.idx_licensor]
    input_to_verb[Verb.idx_dependent] = Noun.state_hist[t, Noun.idx_lex]
    link_nv[t] = (Verb.state_hist[t-1,] @ input_to_verb) / Verb.nfeatures

    Verb.state_hist[t,] = Verb.state_hist[t-1,] + tstep * (-Verb.state_hist[t-1,]
        + np.tanh(Verb.W_rec @ Verb.state_hist[t-1,] + link_nv[t] * input_to_verb))
    verb_sim[t,] = shepard_similarity(Verb.state_hist[t,], verb_patterns.T)
    
    # Introduce the noun at t = 250
    if tvec[t] == 20:
        Noun.state_hist[t,] = noun_patterns[:,1] # phonology for 'dogs'

    # Introduce verb at t = 500
    if tvec[t] == 40:
        Verb.state_hist[t,] = verb_patterns[:,1] # phonology for 'are'


##### Plotting #####
det_labels = ['a', 'these', 'this']
plot_trajectories(tvec, det_sim, det_labels)

noun_labels = ['dog', 'dogs', 'cat', 'cats']
plot_trajectories(tvec, noun_sim, noun_labels)

verb_labels = ['is', 'are']
plot_trajectories(tvec, verb_sim, verb_labels)

plt.plot(tvec, link_dn)
plt.title('Determiner-noun link strength')
plt.show()

plt.plot(tvec, link_nv)
plt.title('Noun-verb link strength')
plt.show()
