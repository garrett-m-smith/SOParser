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

Problem with +=: by the time ti's time to add the next phonological form, the
treelets are basically already at their attractors. The treelet dynamics are
pushing too fast. So using += moves the treelet away from some useful attractor
to some weird mixed state. Attempts to fix:
    1. resecaling recurrent weights (* 1/nfeat). Origin becomes only stable fp.
    2. Setting W_rec diagonal to 0 or 0.1 works!

Possible problem: It's not adding new lexical items that is making
parse formation less stable, it's changing the weight matrices to make the 
different feature banks disconnected. I think this means that I'll need a
separate lexical representation (lexical unit) for each morphological form:
'is', 'are', 'sing', 'sings', etc.

Changes this time around:
    1. Positive link strengths via sigmoid
    2. Similarity using only core dimensions
    3. Introduction of phonological form is now just setting the relevant
    (core) dimensions to the right state instead of +=.
    4. Pseudoinverse method for setting weights: seems to produce reasonable
    results for grammatical and ungrammatical set ups.
    4a. Using the sign of the psinv method does not work!

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
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def scalar_proj(vec, pattern_mat):
    return (vec @ pattern_mat) / np.linalg.norm(pattern_mat, axis = 0)

def shepard_similarity(vec, pattern_mat):
    """Calculate the similarity s(x, y) = exp(-|x - y|**2) (Shepard, 1987)"""
    return np.exp(-np.linalg.norm(pattern_mat - vec, axis = 1)**2)

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
    plt.ylim(0, 1.01)
    plt.legend()
    plt.title('Similarity')
    plt.show()

class Treelet(object):
    def __init__(self, nlex, ndependents, nlicensors, nmorph, dim_names):
        self.nlex = nlex
        self.ndependents = ndependents
        self.nlicensors = nlicensors
        self.nmorph = nmorph
        self.nfeatures = nlex + ndependents + nlicensors + nmorph #len(self.state)
        self.idx = np.arange(self.nfeatures, dtype = 'int')
        self.idx_lex = self.idx[0:self.nlex]
        self.idx_dependent = self.idx[self.nlex:self.nlex + self.ndependents]
        self.idx_licensor = self.idx[self.nlex + self.ndependents:self.nlex + self.ndependents + self.nlicensors]
#        self.idx_lex = np.arange(0, nlex)
#        self.idx_dependent = np.arange(nlex, nlex + ndependents)
#        self.idx_licensor = np.arange(nlex + ndependents, nlex + ndependents + nmorph)
#        self.idx_morph = np.arange(nlex + ndependents + nmorph, self.nfeatures)
        # Kludgy, but...
        self.idx_morph = [-2, -1]
        self.idx_core = np.append(self.idx_lex, self.idx_morph)
        self.state_hist = None
        self.dim_names = dim_names
        self.W_rec = np.zeros((self.nfeatures, self.nfeatures))
    
    def set_recurrent_weights(self, patterns):
#    def set_recurrent_weights(self):
        """Set recurrent weights with inhibitory connections within banks of
        units. Does not yet set weights between feature banks!"""
#        W = np.zeros(self.W_rec.shape)
        W = patterns @ np.linalg.pinv(patterns)
        W[np.ix_(self.idx_lex, self.idx_lex)] = -np.ones((self.nlex, self.nlex))
        if self.ndependents is not 0:
            W[np.ix_(self.idx_dependent, self.idx_dependent)] = -np.ones((self.ndependents, self.ndependents))
            np.fill_diagonal(W[np.ix_(self.idx_dependent, self.idx_dependent)], 1)
        if self.nlicensors is not 0:
            W[np.ix_(self.idx_licensor, self.idx_licensor)] = -np.ones((self.nlicensors, self.nlicensors))
            np.fill_diagonal(W[np.ix_(self.idx_licensor, self.idx_licensor)], 1)
        W[np.ix_(self.idx_morph, self.idx_morph)] = -np.ones((self.nmorph, self.nmorph))
#        np.fill_diagonal(W, 1)
        self.W_rec = W
#        return self.W_rec
        
    def random_initial_state(self, noise_mag):
        noisy_init = np.random.uniform(-noise_mag, noise_mag, self.nfeatures)
        self.set_initial_state(noisy_init)
        
    def set_initial_state(self, vec):
        assert len(vec) == self.nfeatures, 'Wrong length initial state'
        assert self.state_hist is not None, 'state_hist not initialized'
        self.state_hist[0,] = vec
    
    def print_state(self, t = -1):
        for n in range(len(self.dim_names)):
            print('{}:\t{}'.format(self.dim_names[n], self.state_hist[t,n]))
        print('\n')

##### Setting up treelets #####
tstep = 0.01
tvec = np.arange(0.0, 10.0, tstep)

det_patterns = np.array([[1, -1, -1, -1, -1, 0, 0, 0, 0, 1, -1], # a
                         [-1, 1, -1, -1, -1, 0, 0, 0, 0, -1, 1], # these
                         [-1, -1, 1, -1, -1, 0, 0, 0, 0, 1, -1], # this
                         [-1, -1, -1, 1, -1, 0, 0, 0, 0, -1, 1], # many
                         [-1, -1, -1, -1, 1, 0, 0, 0, 0, 1, -1]]).T # each

# Working nouns:
#noun_patterns = np.array([[1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1], # dog
#                          [1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 1], # dogs
#                          [-1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1], # cat
#                          [-1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 1], # cats
#                          [-1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1], # fish
#                          [-1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 1], # fishes
#                          [-1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1], # snail
#                          [-1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1]]).T # snails
#Test
noun_patterns = np.array([[1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, -1], # dog
                          [1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 1], # dogs
                          [-1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 1, -1], # cat
                          [-1, 1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 1], # cats
                          [-1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1], # fish
                          [-1, -1, 1, -1, 0, 0, 0, 0, 0, 0, -1, 1], # fishes
                          [-1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1], # snail
                          [-1, -1, -1, 1, 0, 0, 0, 0, 0, 0, -1, 1]]).T # snails

# Verbs, working:
#verb_patterns = np.array([[1, -1, 0, 0, 0, 0, 1, -1], # is
#                          [-1, 1, 0, 0, 0, 0, -1, 1]]).T # are
# Test:
verb_patterns = np.array([[1, 0, 0, 0, 0, 1, -1], # is
                          [1, 0, 0, 0, 0, -1, 1]]).T # are

# Det treelet
det_dims = ['a', 'these', 'this', 'many', 'each', 'dog', 'cat', 'fish', 'snail', 'sg', 'pl']
Det = Treelet(det_patterns.shape[1], 0, noun_patterns.shape[1]/2, 2, det_dims)
Det.state_hist = np.zeros((len(tvec), Det.nfeatures))
Det.W_rec = det_patterns @ np.linalg.pinv(det_patterns)
#np.fill_diagonal(Det.W_rec, 1)
#Det.set_recurrent_weights(det_patterns)
det_init = det_patterns[:,1]
Det.set_initial_state(det_init)

# Noun treelet:
#noun_dims = ['dog', 'cat', 'fish', 'snail', 'a', 'these', 'this', 'many', 'each', 'is', 'are', 'sg', 'pl']
noun_dims = ['dog', 'cat', 'fish', 'snail', 'a', 'these', 'this', 'many', 'each', 'be', 'sg', 'pl']
Noun = Treelet(noun_patterns.shape[1]/2, det_patterns.shape[1], verb_patterns.shape[1]/2, 2, noun_dims)
Noun.state_hist = np.zeros((len(tvec), Noun.nfeatures))
#Noun.set_recurrent_weights(noun_patterns)
Noun.W_rec = noun_patterns @ np.linalg.pinv(noun_patterns)
#Noun.W_rec[np.ix_(Noun.idx_licensor, Noun.idx_licensor)] = 2
#Noun.set_initial_state(np.random.uniform(-0.01, 0.01, Noun.nfeatures))
Noun.random_initial_state(0.01)

# Verb treelet
#verb_dims = ['is', 'are', 'dog', 'cat', 'fish', 'snail', 'sg', 'pl']
verb_dims = ['be', 'dog', 'cat', 'fish', 'snail', 'sg', 'pl']
Verb = Treelet(verb_patterns.shape[1]/2, noun_patterns.shape[1] / 2, 0, 2, verb_dims)
Verb.state_hist = np.zeros((len(tvec), Verb.nfeatures))
#Verb.set_recurrent_weights(verb_patterns)
Verb.W_rec = verb_patterns @ np.linalg.pinv(verb_patterns)
#np.fill_diagonal(Verb.W_rec, 1)
#Verb.set_initial_state(np.random.uniform(-0.01, 0.01, Verb.nfeatures))
Verb.random_initial_state(0.01)
    


##### Running the whole system #####
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
    link_dn[t] = sigmoid(Det.state_hist[t-1,] @ input_from_n)

    # Determiner dynamics:
    Det.state_hist[t,] = Det.state_hist[t-1,] + tstep * (-Det.state_hist[t-1,]
        + np.tanh(Det.W_rec @ Det.state_hist[t-1,] + link_dn[t] * input_from_n))

    # Calculating the similarity:
    det_sim[t-1,:] = shepard_similarity(Det.state_hist[t, Det.idx_core],det_patterns[Det.idx_core,].T)

    # Noun treelet
    input_from_det = np.zeros(Noun.nfeatures)
    input_from_det[Noun.idx_lex] = Det.state_hist[t-1, Det.idx_licensor]
    input_from_det[Noun.idx_dependent] = Det.state_hist[t-1, Det.idx_lex]
    input_from_det[Noun.idx_morph] = Det.state_hist[t-1, Det.idx_morph]

    input_from_verb = np.zeros(Noun.nfeatures)
    input_from_verb[Noun.idx_licensor] = Verb.state_hist[t-1, Verb.idx_lex]
    input_from_verb[Noun.idx_lex] = Verb.state_hist[t-1, Verb.idx_dependent]
    input_from_verb[Noun.idx_morph] = Verb.state_hist[t-1, Verb.idx_morph]

    Noun.state_hist[t,] = Noun.state_hist[t-1,] + tstep * (-Noun.state_hist[t-1,]
        + np.tanh(Noun.W_rec @ Noun.state_hist[t-1,] + link_dn[t] * input_from_det
                  + link_nv[t] * input_from_verb))
    
    noun_sim[t-1,:] = shepard_similarity(Noun.state_hist[t, Noun.idx_core], noun_patterns[Noun.idx_core,].T)
    
    # Verb treelet
    input_to_verb = np.zeros(Verb.nfeatures)
    input_to_verb[Verb.idx_lex] = Noun.state_hist[t-1, Noun.idx_licensor]
    input_to_verb[Verb.idx_dependent] = Noun.state_hist[t-1, Noun.idx_lex]
    link_nv[t] = sigmoid(Verb.state_hist[t-1,] @ input_to_verb)

    Verb.state_hist[t,] = Verb.state_hist[t-1,] + tstep * (-Verb.state_hist[t-1,]
        + np.tanh(Verb.W_rec @ Verb.state_hist[t-1,] + link_nv[t] * input_to_verb))
    verb_sim[t-1,] = shepard_similarity(Verb.state_hist[t, Verb.idx_core], verb_patterns[Verb.idx_core,].T)
    
    # Introduce noun
    if tvec[t] == 2:
        Noun.state_hist[t,Noun.idx_core] = noun_patterns[Noun.idx_core, 1]

    # Introduce verb
    if tvec[t] == 4:
        Verb.state_hist[t,Verb.idx_core] = verb_patterns[Verb.idx_core, 1]


##### Plotting #####
det_labels = ['a', 'these', 'this', 'many', 'each']
plot_trajectories(tvec, det_sim, det_labels)

noun_labels = ['dog', 'dogs', 'cat', 'cats', 'fish', 'fish[pl]', 'snail', 'snails']
plot_trajectories(tvec, noun_sim, noun_labels)

verb_labels = ['is', 'are']
#verb_labels = ['is', 'are', 'sings', 'sing']
plot_trajectories(tvec, verb_sim, verb_labels)

plt.plot(tvec, link_dn)
plt.title('Determiner-noun link strength')
plt.show()

plt.plot(tvec, link_nv)
plt.title('Noun-verb link strength')
plt.show()

##### Printing final states #####
Det.print_state()
Noun.print_state()
Verb.print_state()