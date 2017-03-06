# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:30:12 2017

@author: garrettsmith

Trying out ANN treelets.


General equations:
    da/dt = -a + f(Wa + I)
    I = link_strength * features_to_match

features_to_match is the vector of features from the treelet at the other end
of the link. Thus, feature passing is implemented as a term in the net
input to the unit, along with the recurrent dynamics (recurrent matrix W,
within-treelet activations a) 

Next Steps:
    1. De-kludgify: re-implement using objects to organize all of the elements
       involved, esp. matching up features across treelets.
    2. Extend noun treelet with PP modifier (and create PP class)
    
Considerations:
    1. How is it handling less than optimal inputs? What states does it end
       up in?
    2. Relatedly, can I define a harmony/energy function for the whole system?
    3. What initial conditions, time constants, etc., make functional parsing
       possible?
"""

import numpy as np
import matplotlib.pyplot as plt


##### Defining utility functions #####
def cosine_similarity(vec, pattern_mat):
    """Calculate cosine similarity between a vector and each column in a 
    matrix. Thus, each pattern should be stored as column vectors in the
    matrix."""
    #assert a.shape == b.shape, "Vectors are of different sizes"
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
    plt.ylabel('Cosine similarity')
    plt.legend()
    plt.title('Cosine similarity')
    plt.show()


##### Setting up treelets #####
# Determiner treelet
# Dimensions: [+a, +some, dog, cat +sg, +pl]
det_patterns = np.array([[1, -1, 0, 0, 1, -1], # a
                         [-1, 1, 0, 0, 0, 0]]).T # some

# Setting weights by hand:
W_det = np.array([[1, -1, 0, 0, 1, -1],
                  [-1, 1, 0, 0, 0, 0],
                  [0, 0, 1, -1, 0, 0],
                  [0, 0, -1, 1, 0, 0],
                  [1, 0, 0, 0, 1, -1],
                  [-1, 0, 0, 0, -1, 1]])    
# Hebbian/covariance matrix for weights
#W_det = (det_patterns @ det_patterns.T) / det_patterns.shape[1]
# Adding noise should eliminate spurious attractors: HKP 91,
# Crisanti & Sompolinsky 1987
#W_det += np.random.uniform(-0.01, 0.01, W_det.shape)
#np.fill_diagonal(W_det, 0)

#det_init = np.array([1, -1, 0, 0]) # activating phonology for 'a'
det_init = np.array([-1, 1, 0, 0, 0, 0]) # activating phonology for 'some'

# Noun treelet
# Dimensions: [+dog, +cat, +a, +some, +sg, +pl]
noun_patterns = np.array([[1, -1, 0, 0, 1, -1], # dog
                          [1, -1, -1, 0, -1, 1], # dogs
                          [-1, 1, 0, 0, 1, -1], # cat
                          [-1, 1, -1, 0, -1, 1]]).T # cats

# Setting weights by hand:
W_noun = np.array([[1, -1, 0, 0, 0, 0],
                   [-1, 1, 0, 0, 0, 0],
                   [0, 0, 1, -1, 1, -1],
                   [0, 0, -1, 1, 0, 0],
                   [0, 0, 1, 0, 1, -1],
                   [0, 0, -1, 0, -1, 1]])
#W_noun = (noun_patterns @ noun_patterns.T) / noun_patterns.shape[1]
#W_noun += np.random.uniform(-0.01, 0.01, W_noun.shape)
#np.fill_diagonal(W_noun, 0)
noun_init = np.random.uniform(-0.001, 0.001, noun_patterns[:,0].shape)

# Verb treelet: is, are, dog, cat, sg, pl
verb_patterns = np.array([[1, -1, 0, 0, 1, -1], # is
                          [-1, 1, 0, 0, -1, 1]]).T # are
W_verb = np.array([[1, -1, 0, 0, 1, -1],
                   [-1, 1, 0, 0, -1, 1],
                   [0, 0, 1, -1, 0, 0],
                   [0, 0, -1, 1, 0, 0],
                   [1, -1, 0, 0, 1, -1],
                   [-1, 1, 0, 0, -1, 1]])
verb_init = np.random.uniform(-0.001, 0.001, verb_patterns[:,0].shape)


##### Running the whole system #####
tvec = np.arange(0.0, 1000.0, 0.1)
det_hist = np.zeros((len(det_init), len(tvec)))
noun_hist = np.zeros((len(noun_init), len(tvec)))
verb_hist = np.zeros((len(verb_init), len(tvec)))
det_hist[:,0] = det_init
noun_hist[:, 0] = noun_init
verb_hist[:, 0] = verb_init

tstep = 0.001
det_sim = np.zeros((len(tvec), det_patterns.shape[1]))
noun_sim = np.zeros((len(tvec), noun_patterns.shape[1]))
verb_sim = np.zeros((len(tvec), verb_patterns.shape[1]))

link_dn = np.zeros(len(tvec))
link_nv = np.zeros(len(tvec))

for t in range(1, len(tvec)):    
    # Still kludge-y

    # Determiner treelet:
    # Noun representation is: dog, cat, a, some, sg., pl.
    # Det representation is:  a, some, dog, cat, sg., pl.
    input_from_n = np.zeros(det_init.shape)
    input_from_n[0:2,] = noun_hist[2:4, t-1]
    input_from_n[2:4,] = noun_hist[0:2, t-1]
    input_from_n[4:,] = noun_hist[4:, t-1]
    
    # Link strength between determiner and noun
    link_dn[t] = (det_hist[:, t-1] @ input_from_n) / len(det_hist[:, t-1])

    # Determiner dynamics:
    det_hist[:, t] = det_hist[:, t-1] + tstep * (-det_hist[:, t-1]
        + np.tanh(W_det @ det_hist[:, t-1] + link_dn[t] * input_from_n))

    # Calculating the similarity:
    det_sim[t,:] = cosine_similarity(det_hist[:, t], det_patterns)

    # Noun treelet
    input_from_det = np.zeros(noun_init.shape)
    input_from_det[0:2,] = det_hist[2:4, t-1]
    input_from_det[2:4,] = det_hist[0:2, t-1]
    input_from_det[4:,] = det_hist[4:, t-1]
    noun_hist[:, t] = noun_hist[:, t-1] + tstep * (-noun_hist[:, t-1] 
        + np.tanh(W_noun @ noun_hist[:, t-1] + link_dn[t] * input_from_det))
    noun_sim[t,:] = cosine_similarity(noun_hist[:, t], noun_patterns)
    
    # Verb treelet
    # Verb representation: is, are, dog, cat, sg., pl.
    # Noun representation is: dog, cat, a, some, sg., pl.
    input_to_verb = np.zeros(verb_init.shape)
    input_to_verb[2:4,] = noun_hist[0:2, t-1]
    input_to_verb[4:,] = noun_hist[4:, t-1]
    link_nv[t] = (verb_hist[:, t-1] @ input_to_verb) / len(verb_hist[:, t-1])
    verb_hist[:, t] = verb_hist[:, t-1] + tstep * (-verb_hist[:, t-1] 
        + np.tanh(W_verb @ verb_hist[:, t-1] + link_nv[t] * input_to_verb))
    verb_sim[t,:] = cosine_similarity(verb_hist[:, t], verb_patterns)
    
    # Introduce the noun at t = 250
    if t == 2500:
        noun_hist[:,t] += np.array([1, -1, 0, 0, -1, 1]) # phonology for 'dogs'
#        noun_hist[:,t] += np.array([1, -1, 0, 0, 1, -1]) # phonology for 'dog'

    # Introduce verb at t = 500
    if t == 5000:
        verb_hist[:,t] += np.array([-1, 1, 0, 0, -1, 1]) # phonology for 'are'
#        verb_hist[:,t] += np.array([1, -1, 0, 0, 1, -1]) # phonology for 'is'


##### Plotting #####
det_labels = ['a', 'some']
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