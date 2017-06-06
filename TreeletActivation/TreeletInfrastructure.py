# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:32:45 2017

@author: garrettsmith

Link dyn. & treelet activations system: Treelet structures

This script will be used to set up Node and Treelet classes with associated
properties, methods, and algorithms.

Notes for future:
    -Link strength is stored as an entry in a dict, while treelet activation
    is an attribute of treelet object. Should change to make consistent.
    -Effect of competition parameter k seems to be different than in other
    LV systems (Frank 2014, Fukai et al. 1997, etc.). WTA might be available
    for k < 1...
    -The treelet activation threshold is another important parameter. 0.1
    seems too low, so need to find a principled way of setting it.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    """Basic structure for nodes/attachment sites on a treelet. Current 
    implementation is kludgy (converting dict to list and then indexing),
    but it works."""
    def __init__(self, attch_dict):
#        self.feature_vector = list(attch_dict.values())
        self.feature_vector = np.array(list(attch_dict.values())[0], dtype = float)
        self.name = list(attch_dict.keys())[0]
        
    def get_feature_vector(self):
        return self.feature_vector


class Treelet(object):
    """A container for all of the nodes in a treelet. Consists of a mother
    node and a dict of daughter nodes."""
    def __init__(self, name, input_dict):
        # daughters arg should be dict of dicts: ea. entry has a name and a
        # feat. vec.
        self.phon_form = input_dict['phon_form']
        self.name = name
        
        # Init activation
        self.activation = 0.01
        
        # Making the mother node
        self.mother = Node({'mother': input_dict['mother_feat']})
        
        # Making the dict for the daughters
        self.daughters = {}
        
        # Adding daughter attachment sites
        print()
        daughters = input_dict['daughters']
        if daughters is not None:
            for attch in daughters:
                curr = {attch: daughters[attch]}
                new_attch = Node(curr)
                self.daughters.update({new_attch.name: new_attch})
    
    def get_activation(self):
        return self.activation
    
    def update_activation(self, new_act):
        self.activation = new_act

class Lexicon(object):
    """Container for Treelets and links, possibly including dynamics. Has
    two parts: a dict of treelets with their associated properties and a dict
    of links. The links dict has three indices (i.e., nested dicts): head
    node on dependent, other (head of phrase) treelet, daughter node on the 
    other treelet."""
    
    def __init__(self):
        # The list of all Treelet objects in the lexicon
        self.treelets = {}
        self.ntreelets = len(self.treelets)
        self.links = {}
        self.nlinks = len(self.treelets)
        self.initialized = False
        self.ntsteps = 0
        self.act_threshold = 0.2
        
    def add_treelet(self, *args):
        """Function for adding a new treelet to the lexicon. Checks if the
        treelet (a phon., mother, daughters pairing) is already in the
        lexicon. Assumes that to-be-added treelet has a unique name."""
        for new_treelet in args:
            if new_treelet.name not in self.treelets:
                self.treelets.update({new_treelet.name: new_treelet})
        self.ntreelets += 1
    
    def _make_links(self):
        """Links are stored in nested dictionaries, so referencing them will
        look like this: 
            lex.links[dependent_treelet][head_treelet][head_daughter]."""
        for dep in self.treelets:
            # Enforcing the no-self-loops constraint
            # Creates a dict of all of the treelets in the lexicon except the
            # current working one
            others = {x: self.treelets[x] for x in self.treelets if x not in dep}
            # Creates from the head of the current treelet (dep) to each 
            # daughter node (attch) on all of the other treelets (head)
            for head in others:
                for attch in self.treelets[head].daughters:
                    self.links.setdefault(dep, {}).setdefault(head, {}).setdefault(attch, {'link_strength': 0, 'feature_match': 0})
                    self.nlinks += 1
    
    def _calc_feature_match(self):
        """Calculates the feature match between the attachemnt sites that
        are connected by a link. Uses NumPy dot() for now."""
        for tr in self.treelets:
            for other_head in {x: self.treelets[x] for x in self.treelets if x not in tr}:
                for attch in self.treelets[other_head].daughters:
                    self.links[tr][other_head][attch]['feature_match'] = \
                    np.dot(self.treelets[tr].mother.feature_vector, 
                           self.treelets[other_head].daughters[attch].feature_vector)
    
    def _import_treelets(self, file):
        """Imports the lexicon from a YAML file."""
        with open(file, 'r') as stream:
            # Imports a dict of dicts
            imported_lex = yaml.safe_load(stream)
        for word in imported_lex:
            # Getting current value of the word
            curr = imported_lex[word]
            new_treelet = Treelet(word, curr)
            self.add_treelet(new_treelet)
            
    def build_lexicon(self, file):
        self._import_treelets(file)
        self._make_links()
        self._calc_feature_match()
        
    def initialize_run(self, ntsteps, init_cond = None):
        """Sets up activation history vectors for treelets and links, and sets
        initial conditions. Right now, can only do random initial conditions."""
        if init_cond is None:
            l0 = np.random.uniform(0, 0.1, size = self.nlinks)
            a0 = np.random.uniform(0, 0.1, size = self.ntreelets)
        for n, treelet in enumerate(self.treelets):
            self.treelets[treelet].activation = np.zeros(ntsteps)
            self.treelets[treelet].activation[0] = a0[n]
        n = -1
        for dep in self.links:
            for head in self.links[dep]:
                for attch in self.links[dep][head]:
                    n += 1
                    self.links[dep][head][attch]['link_strength'] = np.zeros(ntsteps)
                    self.links[dep][head][attch]['link_strength'][0] = l0[n]
        self.initialized = True
        self.ntsteps = ntsteps

    def test_dyn(self):
        assert self.initialized is True, "System has not been initialized."
        for t in range(1, 10):
            for dep in self.links:
                for head in self.links[dep]:
                    for attch in self.links[dep][head]:
                        prev = self.links[dep][head][attch]['link_strength'][t-1]
                        self.links[dep][head][attch]['link_strength'][t] = prev + 1
    
    def get_mother_competitors(self, dep, head, attch, tstep):
        """Returns an np array of the link strengths of the mother-end 
        (dependent) competitors for a link at time tstep. Also multiplies
        them by their link strength."""
        others = [x for x in self.links[dep].keys()]
        vals = []
        for comp in others:
            other_attch = [x for x in self.links[dep][comp].keys()]
            for a_comp in other_attch:
                if comp == head and a_comp == attch:
                    pass
                else:
                    f = self.links[dep][comp][a_comp]['feature_match']
                    vals.append(f * self.links[dep][comp][a_comp]['link_strength'][tstep])
        return np.array(vals)
    
    def get_daughter_competitors(self, dep, head, attch, tstep):
        """Returns an np array of the link strengths of the daughter-end
        (phrase head) competitors for a link at time tstep. Also multiplies
        them by their link strengths."""
        exclude = [dep, head, attch]
        others = [x for x in self.links.keys() if x not in exclude]
        vals = []
        for comp in others:
            f = self.links[comp][head][attch]['feature_match']
            vals.append(f * self.links[comp][head][attch]['link_strength'][tstep])
        return np.array(vals)
    
    def single_run(self, tau, k):
        """Only does link dynamics; no activation dynamics yet."""
        assert self.initialized is True, "System has not been initialized."
        for t in range(1, self.ntsteps):
            # Doing links first
            for dep in self.links:
                for head in self.links[dep]:
                    for attch in self.links[dep][head]:
                        comp_d = self.get_daughter_competitors(dep, head, attch, t-1)
                        comp_a = self.get_mother_competitors(dep, head, attch, t-1)
                        prev = self.links[dep][head][attch]['link_strength'][t-1]
                        curr = prev + tau * (prev 
                                             * (self.treelets[dep].activation[t-1] + self.treelets[head].activation[t-1])
                                             * (1 - prev - k * comp_d.sum()
                                             - k * comp_a.sum()))
                        self.links[dep][head][attch]['link_strength'][t] = curr
            # Next, treelet activations
            for tr in self.treelets:
                prev = self.treelets[tr].activation[t-1]
                # To calculate the average link activation
                coef = 1. / (len(self.treelets[tr].daughters) + 1)
                # Calculate sum of all links attaching to a treelet
                others = [x for x in self.treelets.keys() if x is not tr]
                vals = []
                for incoming in others:
                    # links with tr as dependent
                    if len(self.treelets[incoming].daughters) > 0:
                        for attch in self.links[tr][incoming]:
                            vals.append(self.links[tr][incoming][attch]['link_strength'][t-1])
                    if len(self.treelets[tr].daughters) > 0:
                        daughters = [x for x in self.treelets[tr].daughters.keys()]
                        for d in daughters:
                            vals.append(self.links[incoming][tr][d]['link_strength'][t-1])
                vals = np.array(vals)
                curr = prev + tau * ((-self.act_threshold 
                                      + coef * vals.sum()) * prev
                                     * (1 - prev))
                self.treelets[tr].activation[t] = curr
    
    def plot_traj(self):
        for dep in self.links:
            for head in self.links[dep]:
                for attch in self.links[dep][head]:
                    if self.links[dep][head][attch]['link_strength'][-1] > 0.0:
                        plt.plot(self.links[dep][head][attch]['link_strength'],
                             label = '{}-{}-{}'.format(dep, head, attch))
        plt.legend(bbox_to_anchor = (1.05, 1))
        plt.title('Link strengths')
        plt.show()
        
        for tr in self.treelets:
            plt.plot(self.treelets[tr].activation, label = tr)
        plt.legend()
        plt.title('Treelet activations')
        plt.show()
    
    # Later, possibly add plotting methods from NetworkX to make figures
    # showing the links in the form of a directed graph!


if __name__ == '__main__':
    # Create a lexicon
    lex = Lexicon()
    # Get treelets read in
    lex.build_lexicon('Lexicon.yaml')
    lex.initialize_run(5000)
#    lex.test_dyn()
    lex.single_run(0.01, 2)
    lex.plot_traj()
    
