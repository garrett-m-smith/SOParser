# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:32:45 2017

@author: garrettsmith

Link dyn. & treelet activations system: Treelet structures

This script will be used to set up Node and Treelet classes with associated
properties, methods, and algorithms
"""

import yaml

class Node(object):
    """Basic structure for nodes/attachment sites on a treelet."""
    def __init__(self, attch_dict):
        self.feature_vector = list(attch_dict.values())
        self.name = list(attch_dict.keys())[0]
        
    def get_feature_vector(self):
        return self.feature_vector


class Treelet(object):
    """A container for all of the nodes in a treelet. Consists of a mother
    node and a list of daughter nodes."""
    def __init__(self, name, input_dict):
        # daughters arg should be dict of dicts: ea. entry has a name and a
        # feat. vec.
        self.phon_form = input_dict['phon_form']
        self.name = name
        
        # Init activation & threshold
        self.activation = 0.01
        self.threshold = 0.1
        
        # Making the mother node
        self.mother = Node({'mother': input_dict['mother_feat']})
        
        # Making the dict for the daughters
        self.daughters = {}
        
        # Adding daughter attachment sites
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
    """Container for Treelets and links, possibly including dynamics."""
    
    def __init__(self):
        # The list of all Treelet objects in the lexicon
        self.treelets = {}
        self.ntreelets = len(self.treelets)
        self.links = {}
        self.nlinks = len(self.treelets)
        
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
            lex.links[dependent_treelet][head_treelet][head_daughter].
        It also adds the links to the relevant """
        for dep in self.treelets:
            # Enforcing the no-self-loops constraint
            others = {x: self.treelets[x] for x in self.treelets if x not in dep}
            for head in others:
                for attch in self.treelets[head].daughters:
                    self.links.setdefault(dep, {}).setdefault(head, {}).setdefault(attch, 0)
                    self.nlinks += 1
    
    def _import_treelets(self, file):
        """Imports the lexicon from a YAML file."""
        with open(file, 'r') as stream:
            imported_lex = yaml.safe_load(stream)
        for word in imported_lex:
            curr = imported_lex[word]
            new_treelet = Treelet(word, curr)
            self.add_treelet(new_treelet)
            
    def build_lexicon(self, file):
        self._import_treelets(file)
        self._make_links()
        
    # Later, possibly add plotting methods from NetworkX to make figures
    # showing the links in the form of a directed graph!


if __name__ == '__main__':
#    the = Treelet('the', [1, 0, 1, 0], [])
#    dog = Treelet('dog', [0, 1, 0, 1], [['Det', [1, 0, 1, 0]], ['Adj', [0, 0, 1, 1]]])
#    eats = Treelet('eats', [0, 0, 0, 0], [['Subj', [0, 1, 0, 1]], ['DO', [0, 1, 0, 1]]])
    
    lex = Lexicon()
#    lex.add_treelet(the, dog, eats)
#    lex.make_links()
    lex.build_lexicon('Lexicon.yaml')
    
