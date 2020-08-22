#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


import numpy as np

from MDAnalysis.analysis            import distances            as mdadist 

from numba import jit

import cython

@jit('i8(i4, i8, i4)', nopython=True)
def getLinIndexNumba(src, trgt, n):
    '''Conversion from 2D matrix indices to 1D triangular.
    
    Converts from 2D matrix indices to 1D (n*(n-1)/2) unwrapped triangular matrix index.
    This version of the function is JIT compiled using Numba.
    
    Args:
        src (int): Source node.
        trg (int): Target node.
        n (int): Dimension of square matrix
    
    Returns:
        int: 1D index in unwrapped triangular matrix.
    
    '''
    
    # based on https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    k = (n*(n-1)/2) - (n-src)*((n-src)-1)/2 + trgt - src - 1.0
    return k

@jit('void(i8, i8, f8[:], i8[:], i8[:], i8[:], f8[:])', nopython=True)
def atmToNodeDist(numNodes, nAtoms, tmpDists, atomToNode, nodeGroupIndicesNP, nodeGroupIndicesNPAux, nodeDists):
    '''Translates MDAnalysis distance calculation to node distance matrix.
    
    This function is JIT compiled by Numba to optimize the search for shortest cartesian distances between atoms in different node groups . It relies on the results of MDAnalysis' `self_distance_array` calculation, stored in a 1D NumPy array of shape (n*(n-1)/2,), which acts as an unwrapped triangular matrix.
    
    The pre-allocated triangular matrix passed as an argument to this function is used to store the shortest cartesian distance between each pair of nodes.
    
    This is intended as an analysis tool to allow the comparison of network distances and cartesian distances. It is similar to :py:func:`calcContactC`, which is optimized for contact detection.
    
    Args:
        numNodes (int): Number of nodes in the system.
        nAtoms (int) : Number of atoms in atom groups represented by system nodes. Usually hydrogen atoms are not included in contact detection, and are not present in atom groups.
        atomToNode (obj) : NumPy array that maps atoms in atom groups to their respective nodes.
        nodeGroupIndicesNP (obj) : NumPy array with atom indices for all atoms in each node group.
        nodeGroupIndicesNPAux (obj) : Auxiliary NumPy array with the indices of the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
        nodeDists (obj) : Pre-allocated array to store cartesian distances.
        backend (str) : Controls how MDAnalysis will perform its distance calculations. Options are  `serial` and `openmp`.
        
    '''
    
    maxVal = np.finfo(np.float64).max
    
    # Initialize with maximum possible value of a float64
    tmpDistsAtms = np.full( nAtoms, maxVal, dtype=np.float64 )
    
    # We iterate untill we have only one node left
    for i in range(numNodes - 1):
        
        # Initializes the array for this node.
        tmpDistsAtms.fill(maxVal)
        
        # index of first atom in node "i"
        nodeAtmIndx = nodeGroupIndicesNPAux[i]
        # index of first atom in node "i+1"
        nodeAtmIndxNext = nodeGroupIndicesNPAux[i+1]
        
        # Gets first atom in next node
        nextIfirst = nodeGroupIndicesNP[nodeAtmIndxNext]
        
        # Per node:  Iterate over atoms from node i
        for nodeI_k in nodeGroupIndicesNP[nodeAtmIndx:nodeAtmIndxNext]:
            
            # Go from 2D indices to 1D (n*(n-1)/2) indices:
            j1 = getLinIndexNumba(nodeI_k, nextIfirst, nAtoms)
            jend = j1 + (nAtoms - nextIfirst)
            
            # Gets the shortest distance between atoms in different nodes
            tmpDistsAtms[nextIfirst:] = np.where(tmpDists[j1: jend] < tmpDistsAtms[nextIfirst:], 
                     tmpDists[j1: jend], 
                     tmpDistsAtms[nextIfirst:])
            
        for pairNode in range(i+1, numNodes):
            np.where(atomToNode == pairNode)
            
            # Access the shortests distances between atoms from "pairNode" and the current node "i"
            minDist = np.min(tmpDistsAtms[ np.where(atomToNode == pairNode) ])
            
            # Go from 2D node indices to 1D (numNodes*(numNodes-1)/2) indices:
            ijLI = getLinIndexNumba(i, pairNode, numNodes)
            
            nodeDists[ijLI] = minDist
    
# High memory usage (nAtoms*(nAtoms-1)/2), calcs all atom distances at once.
# We use self_distance_array and iterate over the trajectory.
# https://www.mdanalysis.org/mdanalysis/documentation_pages/analysis/distances.html
def calcDistances(selection, numNodes, nAtoms, atomToNode, 
                       nodeGroupIndicesNP, nodeGroupIndicesNPAux, nodeDists, backend="serial" ):
    '''Executes MDAnalysis atom distance calculation and node cartesian distance calculation.
    
    This function is a wrapper for two optimized distance calculation and node distance calculation calls.
    The first is MDAnalysis' `self_distance_array`. The second is the internal :py:func:`atmToNodeDist`.
    All results are stored in pre-allocated NumPy arrays.
    
    This is intended as an analysis tool to allow the comparison of network distances and cartesian distances. It is similar to :py:func:`getContactsC`, which is optimized for contact detection.
    
    Args:
        selection (str) : Atom selection for the system being analyzed.
        numNodes (int): Number of nodes in the system.
        nAtoms (int) : Number of atoms in atom groups represented by system nodes. Usually hydrogen atoms are not included in contact detection, and are not present in atom groups.
        atomToNode (obj) : NumPy array that maps atoms in atom groups to their respective nodes.
        nodeGroupIndicesNP (obj) : NumPy array with atom indices for all atoms in each node group.
        nodeGroupIndicesNPAux (obj) : Auxiliary NumPy array with the indices of the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
        nodeDists (obj) : Pre-allocated array to store cartesian distances.
        backend (str) : Controls how MDAnalysis will perform its distance calculations. Options are  `serial` and `openmp`.
        
    '''
    
    
    tmpDists = np.zeros( int(nAtoms*(nAtoms-1)/2), dtype=np.float64 )
    
    # serial vs OpenMP
    mdadist.self_distance_array(selection.positions, result=tmpDists, backend=backend)
    
    # Translate atoms distances in minimum node distance.
    atmToNodeDist(numNodes, nAtoms, tmpDists, atomToNode, nodeGroupIndicesNP, nodeGroupIndicesNPAux, nodeDists)

@cython.cfunc
@cython.returns(cython.int)
@cython.locals(src=cython.int, trgt=cython.int, n=cython.int)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getLinIndexC(src, trgt, n):
    '''Conversion from 2D matrix indices to 1D triangular.
    
    Converts from 2D matrix indices to 1D (n*(n-1)/2) unwrapped triangular matrix index.
    This version of the function is compiled using Cython.
    
    Args:
        src (int): Source node.
        trg (int): Target node.
        n (int): Dimension of square matrix
    
    Returns:
        int: 1D index in unwrapped triangular matrix.
    
    '''
    
    # based on https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    k = (n*(n-1)/2) - (n-src)*((n-src)-1)/2 + trgt - src - 1.0
    return int(k)
    
@cython.cfunc
@cython.returns(cython.void)
@cython.locals(numNodes=cython.int, nAtoms=cython.int, cutoffDist=cython.float,
               tmpDists="np.ndarray[np.float_t, ndim=1]",
               tmpDistsAtms="np.ndarray[np.float_t, ndim=1]",
               contactMat="np.ndarray[np.int_t, ndim=2]",
               atomToNode="np.ndarray[np.int_t, ndim=1]",
               nodeGroupIndicesNP="np.ndarray[np.int_t, ndim=1]",
               nodeGroupIndicesNPAux="np.ndarray[np.int_t, ndim=1]")
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calcContactC(numNodes, nAtoms, cutoffDist, 
                 tmpDists, 
                 tmpDistsAtms, 
                 contactMat,
                 atomToNode,
                 nodeGroupIndicesNP,
                 nodeGroupIndicesNPAux):
    '''Translates MDAnalysis distance calculation to node contact matrix.
    
    This function is Cython compiled to optimize the search for nodes in contact. It relies on the results of MDAnalysis' `self_distance_array` calculation, stored in a 1D NumPy array of shape (n*(n-1)/2,), which acts as an unwrapped triangular matrix.
    
    In this function, the distances between all atoms in an atom groups of all pairs of nodes are verified to check if any pair of atoms were closer than a cutoff distance. This is done for all pairs of nodes in the system, and all frames in the trajectory. The pre-allocated contact matrix passed as an argument to this function is used to store the number of frames where each pair of nodes had at least one contact.
    
    Args:
        numNodes (int): Number of nodes in the system.
        nAtoms (int) : Number of atoms in atom groups represented by system nodes. Usually hydrogen atoms are not included in contact detection, and are not present in atom groups.
        cutoffDist (float) : Distance at which atoms are no longer considered 'in contact'.
        tmpDists (obj) : Temporary pre-allocated NumPy array with atom distances. This is the result of MDAnalysis `self_distance_array` calculation.
        tmpDistsAtms (obj) : Temporary pre-allocated NumPy array to store the shortest distance between atoms in different nodes.
        contactMat (obj) : Pre-allocated NumPy matrix where node contacts will be stored.
        atomToNode (obj) : NumPy array that maps atoms in atom groups to their respective nodes.
        nodeGroupIndicesNP (obj) : NumPy array with atom indices for all atoms in each node group.
        nodeGroupIndicesNPAux (obj) : Auxiliary NumPy array with the indices of the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
    
    '''
    
    #cdef int nextIfirst, j1, jend, i, nodeI_k, nodeAtmIndx, nodeAtmIndxNext
    nextIfirst = cython.declare(cython.int)
    
    # Cython types are evaluated as for cdef declarations
    j1: cython.int
    jend: cython.int
    i: cython.int
    nodeI_k: cython.int
    nodeAtmIndx: cython.int
    nodeAtmIndxNext: cython.int
    
    
    # We iterate untill we have only one node left
    for i in range(numNodes - 1):
        
        # Initializes the array for this node.
        tmpDistsAtms.fill(cutoffDist*2)
        
        # index of first atom in node "i"
        nodeAtmIndx = nodeGroupIndicesNPAux[i]
        # index of first atom in node "i+1"
        nodeAtmIndxNext = nodeGroupIndicesNPAux[i+1]
        
        # Gets first atom in next node
        nextIfirst = nodeGroupIndicesNP[nodeAtmIndxNext]
        
        # Per node:  Iterate over atoms from node i
        for nodeI_k in nodeGroupIndicesNP[nodeAtmIndx:nodeAtmIndxNext]:
            
            # Go from 2D indices to 1D (n*(n-1)/2) indices:
            j1 = getLinIndexC(nodeI_k, nextIfirst, nAtoms)
            jend = j1 + (nAtoms - nextIfirst)
            
            # Gets the shortest distance between atoms in different nodes
            tmpDistsAtms[nextIfirst:] = np.where(tmpDists[j1: jend] < tmpDistsAtms[nextIfirst:], 
                     tmpDists[j1: jend], 
                     tmpDistsAtms[nextIfirst:])
        
        # Adds one to the contact to indicate that this frame had a contact.
        contactMat[i, np.unique( atomToNode[ np.where(tmpDistsAtms < cutoffDist)[0] ] )] += 1

# High memory usage (nAtoms*(nAtoms-1)/2), calcs all atom distances at once.
@cython.cfunc
@cython.returns(cython.void)
@cython.locals(numNodes=cython.int, nAtoms=cython.int, cutoffDist=cython.float,
               tmpDists="np.ndarray[np.float_t, ndim=1]",
               tmpDistsAtms="np.ndarray[np.float_t, ndim=1]",
               contactMat="np.ndarray[np.int_t, ndim=2]",
               atomToNode="np.ndarray[np.int_t, ndim=1]",
               nodeGroupIndicesNP="np.ndarray[np.int_t, ndim=1]",
               nodeGroupIndicesNPAux="np.ndarray[np.int_t, ndim=1]")
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getContactsC(selection, numNodes, 
                        nAtoms, 
                        cutoffDist, 
                        tmpDists, 
                        tmpDistsAtms, 
                        contactMat,
                        atomToNode,
                        nodeGroupIndicesNP,
                        nodeGroupIndicesNPAux):
    '''Executes MDAnalysis atom distance calculation and node contact detection.
    
    This function is Cython compiled as a wrapper for two optimized distance calculation and contact determination calls.
    The first is MDAnalysis' `self_distance_array`. The second is the internal :py:func:`calcContactC`.
    All results are stored in pre-allocated NumPy arrays.
    
    
    Args:
        selection (str) : Atom selection for the system being analyzed.
        numNodes (int): Number of nodes in the system.
        nAtoms (int) : Number of atoms in atom groups represented by system nodes. Usually hydrogen atoms are not included in contact detection, and are not present in atom groups.
        cutoffDist (float) : Distance at which atoms are no longer considered 'in contact'.
        tmpDists (obj) : Temporary pre-allocated NumPy array with atom distances. This is the result of MDAnalysis `self_distance_array` calculation.
        tmpDistsAtms (obj) : Temporary pre-allocated NumPy array to store the shortest distance between atoms in different nodes.
        contactMat (obj) : Pre-allocated NumPy matrix where node contacts will be stored.
        atomToNode (obj) : NumPy array that maps atoms in atom groups to their respective nodes.
        nodeGroupIndicesNP (obj) : NumPy array with atom indices for all atoms in each node group.
        nodeGroupIndicesNPAux (obj) : Auxiliary NumPy array with the indices of the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
    
    '''
    
    # serial vs OpenMP
    mdadist.self_distance_array(selection.positions, result=tmpDists, backend='openmp')
    
    calcContactC(numNodes, nAtoms, cutoffDist, tmpDists, tmpDistsAtms, 
                 contactMat, atomToNode, nodeGroupIndicesNP, nodeGroupIndicesNPAux)


