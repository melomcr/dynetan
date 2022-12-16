#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


from networkx.algorithms.shortest_paths.dense import floyd_warshall as nxFW
from networkx.algorithms.shortest_paths.dense import floyd_warshall_predecessor_and_distance as nxFWPD
from networkx                       import edge_betweenness_centrality as nxbetweenness

from collections    import OrderedDict

import copy
import numpy as np

def calcOptPathPar(nxGraphs, inQueue, outQueue):
    '''Wrapper to calculate Floyd Warshall optimal path in parallel.
    
    For the FW optimal path determination, we use the node "distance" as weights, 
    that is, the log-transformation of the correlations, NOT the correlation itself.
    
    For every window, the function stores in the output queue the optimal paths. It turns the dictionary of distances returned by NetworkX into a NumPy 2D array per window of trajectory, which allows significant speed up in data analysis.
    
    Args:
        nxGraphs (obj) : NetworkX object.
        inQueue (obj) : Multiprocessing queue object for acquiring jobs.
        outQueue (obj) : Multiprocessing queue object for placing results.
    
    '''
    
    while not inQueue.empty():
        
        win = inQueue.get()
        
        ### IMPORTANT!
        # For the FW optimal path determination, we use the "distance" as weight, 
        #  that is, the log-transformation of the correlations. NOT the correlation itself.
        pathsPred, pathsDist = nxFWPD(nxGraphs[win], weight='dist')

        # Turns dictionary of distances into NumPy 2D array per window (for speed up)
        # Notice the nested list comprehensions.
        dists = np.array([[pathsDist[i][j] for i in sorted(pathsDist[j])] for j in sorted(pathsDist)])

        outQueue.put( (win, dists, copy.deepcopy(pathsPred)) )

def calcBetweenPar(nxGraphs, inQueue, outQueue):
    '''Wrapper to calculate betweeness in parallel.
    
    The betweeness calculations used here only take into account the number of paths 
    passing through a given edge, so no weight are considered.
    
    For every window, the function stores in the output queue an ordered dict of node pairs with betweenness higher than zero.
    
    Args:
        nxGraphs (obj) : NetworkX object.
        inQueue (obj) : Multiprocessing queue object for acquiring jobs.
        outQueue (obj) : Multiprocessing queue object for placing results.
    
    '''
    
    while not inQueue.empty():
        
        win = inQueue.get()
        
        # Calc all betweeness in entire system.
        ### IMPORTANT!
        # For the betweeness, we only care about the number of shortests paths 
        #   passing through a given edge, so no weight are considered.
        btws = nxbetweenness(nxGraphs[win], weight=None)
        
        # Creates an ordered dict of pairs with betweenness higher than zero.
        btws = {k:btws[k] for k in btws.keys() if btws[k] > 0}
        btws = OrderedDict(sorted(btws.items(), key=lambda t: t[1], reverse=True))
        
        outQueue.put((win, btws))
