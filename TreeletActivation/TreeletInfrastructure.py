# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:32:45 2017

@author: garrettsmith

Link dyn. & treelet activations system: Treelet structures

This script will be used to set up Node and Treelet classes with associated
properties, methods, and algorithms
"""

class Node(object):
    """Basic structure for nodes/attachment sites on a treelet."""
    def __init__(self, attch_name, feature_vector):
        self.feature_vector = feature_vector
        # Links should be a list of lists 
        self.links = []
        # Give attch. site name, z.B., 'DO', 'mother'
        self.name = attch_name
        
    def get_feature_vector(self):
        return self.feature_vector
    
    def add_link(self, mother, daughter_treelet, daughter_site):
        """Indexing scheme: mother treelet, daughter treelet, daughter
        attachment site."""
        self.links.append([mother, daughter_treelet, daughter_site])


class Treelet(object):
    """A container for all of the nodes in a treelet. Consists of a mother
    node and a list of daughter nodes."""
    def __init__(self, phon_form, mother_feat, daughters):
        # daughters arg should be list of lists: ea. entry has a name and a
        # feat. vec.
        self.phon_form = phon_form
        self.name = phon_form
        
        # Init activation & threshold
        self.activation = 0.01
        self.threshold = 0.1
        
        # Making the mother node
        self.mother = Node('mother', mother_feat)
        
        # Making the list for the daughters
        self.daughters = {}
        
        # Adding daughter attachment sites
        for attch in range(len(daughters)):
            curr = daughters[attch]
            new_attch = Node(curr[0], curr[1])
            self.daughters.update({new_attch.name: new_attch})
    
    def get_activation(self):
        return self.activation
    
    def update_activation(self, new_act):
        self.activation = new_act


class Lexicon(object):
    """Container for Treelets and links, including dynamics."""
    
    def __init__(self):
        # The list of all Treelet objects in the lexicon
        self.treelets = {}
        self.ntreelets = len(self.treelets)
        self.links = {}
        
    def add_treelet(self, *args):
        """Function for adding a new treelet to the lexicon. Checks if the
        treelet (a phon., mother, daughters pairing) is already in the
        lexicon. Assumes that to-be-added treelet has a unique name."""
        for new_treelet in args:
            if new_treelet.phon_form not in self.treelets:
                self.treelets.update({new_treelet.phon_form: new_treelet})
    
    def make_links(self):
        """Links are stored in nested dictionaries, so referencing them will
        look like this: 
            lex.links[dependent_treelet][head_treelet][head_daughter].
        It also adds the links to the relevant """
        for dep in self.treelets:
            # Enforcing the no-self-loops constraint
            others = {x: self.treelets[x] for x in self.treelets if x not in dep}
            for head in others:
                for attch in self.treelets[head].daughters:
                    self.links.setdefault(dep, {}).setdefault(head, {}).setdefault(attch, 0)
    
    def import_treelets(self):
        


if __name__ == '__main__':
    the = Treelet('the', [1, 0, 1, 0], [])
    dog = Treelet('dog', [0, 1, 0, 1], [['Det', [1, 0, 1, 0]], ['Adj', [0, 0, 1, 1]]])
    eats = Treelet('eats', [0, 0, 0, 0], [['Subj', [0, 1, 0, 1]], ['DO', [0, 1, 0, 1]]])
    
    lex = Lexicon()
    lex.add_treelet(the, dog, eats)
    lex.make_links()
    