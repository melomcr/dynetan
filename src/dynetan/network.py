#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr

from networkx.algorithms.shortest_paths.dense import \
    floyd_warshall_predecessor_and_distance as nxFWPD
from networkx import edge_betweenness_centrality as nxbetweenness

from collections import OrderedDict
import copy
import numpy as np
from multiprocessing import Queue
import queue  # So that we can catch the exceptions for empty queues
import networkx as nx
from typing import List


def calcOptPathPar(nx_graphs: List[nx.Graph],
                   in_queue: Queue,
                   out_queue: Queue) -> None:
    """Wrapper to calculate Floyd Warshall optimal path in parallel.

    For the FW optimal path determination, we use the node "distance" as weights,
    that is, the log-transformation of the correlations, NOT the correlation itself.

    For every window, the function stores in the output queue the optimal paths.
    It turns the dictionary of distances returned by NetworkX into a NumPy 2D
    array per window of trajectory, which allows significant speed up in data analysis.

    Args:
        nx_graphs (Any) : NetworkX graph object.
        in_queue (Queue) : Multiprocessing queue object for acquiring jobs.
        out_queue (Queue) : Multiprocessing queue object for placing results.

    """

    # While we still have elements to process, get a new window.
    # We verify if we run out of elements by catching a termination flag
    # in the queue. There is at least one termination flag per initiated process.
    while True:

        try:
            win = in_queue.get(block=True, timeout=0.01)
        except queue.Empty:  # pragma: no cover
            continue
            # If we need to wait for longer for a new item,
            # just continue the loop
        else:
            # The termination flag is an invalid window index of -1.
            if win == -1:
                break

            # IMPORTANT! ###
            # For the FW optimal path determination, we use the "distance" as weight,
            #  that is, the log-transformation of the correlations. NOT the
            #  correlation itself.
            pathsPred, pathsDist = nxFWPD(nx_graphs[win], weight='dist')

            # Turns dictionary of distances into NumPy 2D array per window (for speed up)
            # Notice the nested list comprehensions.
            dists = np.array([[pathsDist[i][j]
                               for i in sorted(pathsDist[j])]
                              for j in sorted(pathsDist)])

            out_queue.put((win, dists, copy.deepcopy(pathsPred)))


def calcBetweenPar(nx_graphs: List[nx.Graph],
                   in_queue: Queue,
                   out_queue: Queue) -> None:
    """Wrapper to calculate betweenness in parallel.

    The betweenness calculations used here only take into account the number of
    paths passing through a given edge, so no weight are considered.

    For every window, the function stores in the output queue an ordered dict
    of node pairs with betweenness higher than zero.

    Args:
        nx_graphs (Any) : NetworkX graph object.
        in_queue (Queue) : Multiprocessing queue object for acquiring jobs.
        out_queue (Queue) : Multiprocessing queue object for placing results.

    """

    # While we still have elements to process, get a new window.
    # We verify if we run out of elements by catching a termination flag
    # in the queue. There is at least one termination flag per initiated process.
    while True:

        try:
            win = in_queue.get(block=True, timeout=0.01)
        except queue.Empty:  # pragma: no cover
            continue
            # If we need to wait for longer for a new item,
            # just continue the loop
        else:
            # The termination flag is an invalid window index of -1.
            if win == -1:
                break

            # Calc all betweenness values for all edges in entire system.
            # IMPORTANT! ###
            # For the betweenness, we only care about the number of the shortest
            # paths passing through a given edge, so no weight are considered.
            btws = nxbetweenness(nx_graphs[win], weight=None)

            # Creates an ordered dict of pairs with betweenness higher than zero.
            btws = {k: btws[k] for k in btws.keys() if btws[k] > 0}
            btws = OrderedDict(sorted(btws.items(), key=lambda t: t[1], reverse=True))

            out_queue.put((win, btws))
