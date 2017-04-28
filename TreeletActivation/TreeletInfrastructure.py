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
        self.attch_name = attch_name
        
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
        
        # Making the mother node
        self.mother = Node('mother', mother_feat)
        
        # Making the list for the daughters
        self.daughters = []
        
        # Adding daughter attachment sites
        for (attch in daughters):
            self.daughters.append(Node(attch[0], attch[1]))