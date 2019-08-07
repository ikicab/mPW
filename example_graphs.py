# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

import networkx as nx
import igraph as ig
from scipy.io import mmread
import codecs
import numpy as np
import pickle

from collections import defaultdict
from itertools import count

def hashing(seq):
    """
    Maps a sequence of labels to a sequence of consecutive natural numbers
    (starting from 0).
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    seq : a sequence of labels
    """

    mapping = defaultdict(count().__next__)

    return [mapping[el] for el in seq]

###############################################################################
### Zachary's karate club
###############################################################################
g_nx_karate = nx.karate_club_graph()

g_karate = ig.Graph(directed=False)
g_karate.add_vertices(g_nx_karate.nodes())
g_karate.add_edges(g_nx_karate.edges())

for i in range(len(g_nx_karate.nodes)):
    g_karate.vs[i]['cluster'] = g_nx_karate.nodes[i]['club']

keys = g_karate.vs['cluster']
values = hashing(keys)
dictionary = dict(zip(keys, values))

for v in g_karate.vs:
    v['cluster'] = dictionary[v['cluster']]

###############################################################################
### American College football
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
# http://plato.tp.ph.ic.ac.uk/~time/networks/resources.html
###############################################################################
g_football_old = ig.Graph.Read_GML('datasets/football.gml')
g_football_old.vs['cluster'] = g_football_old.vs['value']

# Corrected dataset
g_football = ig.Graph.Read_GML('datasets/footballTSEinput.gml')
g_football.vs['cluster'] = g_football.vs['value']

###############################################################################
### Dolphins
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
###############################################################################
g_dolphins = ig.Graph.Read_GML('datasets/dolphins.gml')

for v in g_dolphins.vs:
    if v['label'] in ['DN63', 'Knit', 'Beescratch', 'SN90', 'TR82', 'Upbang', 'Number1', 'Notch', 'Web', 'Jet', 'Mus',
                     'Gallatin', 'DN16', 'Feather', 'DN21', 'Quasi', 'Wave', 'MN23', 'Ripplefluke', 'Zig']:
        v['cluster'] = 1
    else:
        v['cluster'] = 0

###############################################################################
### Jazz musicians
#------------------------------------------------------------------------------
# http://deim.urv.cat/~alexandre.arenas/data/welcome.htm
###############################################################################
g_jazz = ig.read('datasets/jazz.net', format='pajek')
g_jazz.simplify()

###############################################################################
### E-mail University Rovira i Virgili
#------------------------------------------------------------------------------
# http://deim.urv.cat/~alexandre.arenas/data/welcome.htm
###############################################################################
g_mailURV = ig.Graph.Read_Ncol('datasets/email.txt')
g_mailURV = g_mailURV.as_undirected()
g_mailURV.simplify()

###############################################################################
### UK faculty
#------------------------------------------------------------------------------
# http://hal.elte.hu/~nepusz/research/datasets/
###############################################################################
g_UKfaculty = ig.Graph.Read_GraphML('datasets/univ_dataset_TSPE.graphml')
g_UKfaculty.vs['cluster'] = g_UKfaculty.vs['group']
g_UKfaculty = g_UKfaculty.as_undirected()

###############################################################################
### US airports
###############################################################################
#R
#rm(list=ls())
#library(igraph)
#library(igraphdata)
#data(package="igraphdata")
#data(USairports)
#write.graph(USairports, "USairports.gml", "gml")
##summary(USairports)

with open('datasets/g_USairports.pkl', 'rb') as f:
    g_USairports = pickle.load(f)

###############################################################################
### Coautorships in condensed matter physics
#------------------------------------------------------------------------------
# https://www.cise.ufl.edu/research/sparse/matrices/Newman/cond-mat-2005.html
###############################################################################
g_coautorships = ig.Graph.Read_GML('datasets/cond-mat-2005.gml')
g_coautorships = g_coautorships.clusters().giant()

###############################################################################
### High-energy theory collaborations
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
###############################################################################
g_hetcol = ig.Graph.Read_GML('datasets/hep-th.gml')
g_hetcol = g_hetcol.clusters().giant()

###############################################################################
### Yeast
#------------------------------------------------------------------------------
# https://sparse.tamu.edu/Pajek/yeast
# http://vlado.fmf.uni-lj.si/pub/networks/data/bio/Yeast/Yeast.htm
###############################################################################
g_yeast = ig.Graph.Read_Edgelist('datasets/yeast.mtx')
g_yeast.delete_vertices(0)
g_yeast = g_yeast.as_undirected()
g_yeast.simplify()

clusters = mmread('datasets/yeast_PIN_class.mtx').tolist()
g_yeast.vs['cluster'] = [c[0] for c in clusters]
g_yeast = g_yeast.clusters().giant()

###############################################################################
### Notre Dame web
#------------------------------------------------------------------------------
# https://snap.stanford.edu/data/web-NotreDame.html
###############################################################################
g_webND = ig.read('datasets/web-NotreDame.txt', format='edge')
g_webND = g_webND.as_undirected()

###############################################################################
### E-mail EU-core
#------------------------------------------------------------------------------
# https://snap.stanford.edu/data/email-Eu-core.html
###############################################################################
g_emailEU = ig.read('datasets/email-Eu-core.txt', format='edge')
g_emailEU = g_emailEU.as_undirected()

file = codecs.open('datasets/email-Eu-core-department-labels.txt')
file_data = np.loadtxt(file, usecols=(0,1), dtype=int)

clusters = file_data[:,1]
g_emailEU.vs['cluster'] = clusters
g_emailEU = g_emailEU.clusters().giant()

###############################################################################
### Coauthorships in network science
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
###############################################################################
g_netsci = ig.Graph.Read_GML('datasets/netscience.gml')
g_netsci = g_netsci.clusters().giant()

###############################################################################
### Political blogs
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
###############################################################################
g_polblogs =  ig.Graph.Read_GML('datasets/polblogs.gml')

for v in g_polblogs.vs:
    v['cluster'] = v['value']

g_polblogs = g_polblogs.as_undirected()
g_polblogs = g_polblogs.clusters().giant()
g_polblogs.simplify()

###############################################################################
### Political books
#------------------------------------------------------------------------------
# http://www-personal.umich.edu/~mejn/netdata/
###############################################################################
g_polbooks =  ig.Graph.Read_GML('datasets/polbooks.gml')
    
for v in g_polbooks.vs:
    v['cluster'] = v['value']

keys = g_polbooks.vs['cluster']
values = hashing(keys)
dictionary = dict(zip(keys, values))

for v in g_polbooks.vs:
    v['cluster'] = dictionary[v['cluster']]

###############################################################################
### Cora citation network
#------------------------------------------------------------------------------
# http://konect.uni-koblenz.de/networks/subelj_cora
###############################################################################
g_cora = ig.Graph.Read_Edgelist('datasets/out.subelj_cora_cora')
g_cora.delete_vertices(0)
g_cora = g_cora.as_undirected()

file = codecs.open('datasets/ent.subelj_cora_cora.class.name')
file_data = np.loadtxt(file, dtype=str)

g_cora.vs['cluster'] = file_data

keys = g_cora.vs['cluster']
values = hashing(keys)
dictionary = dict(zip(keys, values))

for v in g_cora.vs:
    v['cluster'] = dictionary[v['cluster']]

###############################################################################
### International E-road network
#------------------------------------------------------------------------------
# http://konect.uni-koblenz.de/networks/subelj_euroroad
###############################################################################
with open('datasets/g_euroroad.pkl', 'rb') as f: 
    g_euroroad = pickle.load(f)