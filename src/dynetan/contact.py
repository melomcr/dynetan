#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


import numpy as np
import numpy.typing as npt
import MDAnalysis as mda

from MDAnalysis.analysis import distances as mdadist
from MDAnalysis.lib import distances as mdalibdist
from numba import jit
from numba import prange

# For timing and benchmarks
from timeit import default_timer as timer
from datetime import timedelta

from typing import Any

MODE_ALL    = 0  # noqa:E221
MODE_CAPPED = 1


@jit('i8(i8, i8, i8)', nopython=True, nogil=True, cache=True)
def get_lin_index_numba(src: int, trgt: int, n: int) -> int:  # pragma: no cover
    """Conversion from 2D matrix indices to 1D triangular.

    Converts from 2D matrix indices to 1D (n*(n-1)/2) unwrapped triangular
    matrix index. This version of the function is JIT compiled using Numba.

    Args:
        src (int): Source node.
        trgt (int): Target node.
        n (int): Dimension of square matrix

    Returns:
        int: 1D index in unwrapped triangular matrix.

    """
    # PyTest-cov does not detect test coverage over JIT compiled Numba functions

    # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    k = (n*(n-1)/2) - (n-src)*((n-src)-1)/2 + trgt - src - 1.0
    return int(k)


@jit('void(i8, i8, f8[:], i8[:], i8[:], i8[:], f8[:])', nopython=True, parallel=True)
def atm_to_node_dist(num_nodes: int,
                     n_atoms: int,
                     tmp_dists: npt.NDArray[np.float64],
                     atom_to_node: npt.NDArray[np.int64],
                     node_group_indices_np: npt.NDArray[np.int64],
                     node_group_indices_np_aux: npt.NDArray[np.int64],
                     node_dists: npt.NDArray[np.float64]) -> None:  # pragma: no cover
    """Translates MDAnalysis distance calculation to node distance matrix.

    This function is JIT compiled by Numba to optimize the search for shortest
    cartesian distances between atoms in different node groups . It relies on
    the results of MDAnalysis' `self_distance_array` calculation, stored in a 1D
    NumPy array of shape (n*(n-1)/2,), which acts as an unwrapped triangular matrix.

    The pre-allocated triangular matrix passed as an argument to this function
    is used to store the shortest cartesian distance between each pair of nodes.

    This is intended as an analysis tool to allow the comparison of network
    distances and cartesian distances. It is similar to :py:func:`calc_contact_c`,
    which is optimized for contact detection.

    Args:
        num_nodes (int): Number of nodes in the system.
        n_atoms (int) : Number of atoms in atom groups represented by system nodes.
            Usually hydrogen atoms are not included in contact detection, and
            are not present in atom groups.
        tmp_dists (Any) : Temporary pre-allocated NumPy array with atom distances.
            This is the result of MDAnalysis `self_distance_array` calculation.
        atom_to_node (Any) : NumPy array that maps atoms in atom groups to their
            respective nodes.
        node_group_indices_np (Any) : NumPy array with atom indices for all atoms
            in each node group.
        node_group_indices_np_aux (Any) : Auxiliary NumPy array with the indices of
            the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
        node_dists (Any) : Pre-allocated array to store cartesian distances
            between *nodes*. This is a linearized upper triangular matrix.

    """
    # PyTest-cov does not detect test coverage over JIT compiled Numba functions

    max_val: Any = np.finfo(np.float64).max

    # Initialize with maximum possible value of a float64
    tmp_dists_atms: npt.NDArray[np.float64] = np.full(n_atoms, max_val, dtype=np.float64)

    # We iterate until we have only one node left
    for i in range(num_nodes - 1):

        # Initializes the array for this node.
        tmp_dists_atms.fill(max_val)

        # index of first atom in node "i"
        node_atm_indx = node_group_indices_np_aux[i]
        # index of first atom in node "i+1"
        node_atm_indx_next = node_group_indices_np_aux[i + 1]

        # Gets first atom in next node
        next_i_first = node_group_indices_np[node_atm_indx_next]

        # Per node:  Iterate over atoms from node i
        for node_i_k in node_group_indices_np[node_atm_indx:node_atm_indx_next]:

            # Go from 2D indices to 1D (n*(n-1)/2) indices:
            j1 = get_lin_index_numba(node_i_k, next_i_first, n_atoms)
            jend = j1 + (n_atoms - next_i_first)

            # Gets the shortest distance between atoms in different nodes
            tmp_dists_atms[next_i_first:] = np.where(
                tmp_dists[j1: jend] < tmp_dists_atms[next_i_first:],
                tmp_dists[j1: jend],
                tmp_dists_atms[next_i_first:])

        for pairNode in prange(i+1, num_nodes):

            # Access the shortest distances between atoms from "pairNode" and
            # the current node "i"
            minDist = np.min(tmp_dists_atms[np.where(atom_to_node == pairNode)])

            # Go from 2D node indices to 1D (numNodes*(numNodes-1)/2) indices:
            ijLI = get_lin_index_numba(i, pairNode, num_nodes)

            node_dists[ijLI] = minDist


# High memory usage (nAtoms*(nAtoms-1)/2), calcs all atom distances at once.
# We use self_distance_array and iterate over the trajectory.
# https://www.mdanalysis.org/mdanalysis/documentation_pages/analysis/distances.html
def calc_distances(selection: mda.AtomGroup,
                   num_nodes: int,
                   n_atoms: int,
                   atom_to_node: npt.NDArray[np.int64],
                   cutoff_dist: float,
                   node_group_indices_np: npt.NDArray[np.int64],
                   node_group_indices_np_aux: npt.NDArray[np.int64],
                   node_dists: npt.NDArray[np.float64],
                   backend: str = "serial",
                   dist_mode: int = MODE_ALL,
                   verbose: int = 0) -> None:
    """Executes MDAnalysis atom distance calculation and node cartesian
    distance calculation.

    This function is a wrapper for two optimized atomic distance calculation
    and node distance calculation calls. The first is one of MDAnalysis' atom
    distance calculation functions (either `self_distance_array` or
    `self_capped_distance`). The second is the internal :py:func:`atm_to_node_dist`.
    All results are stored in pre-allocated NumPy arrays.

    This is intended as an analysis tool to allow the comparison of network
    distances and cartesian distances. It is similar to :py:func:`get_contacts_c`,
    which is optimized for contact detection.

    Args:
        selection (str) : Atom selection for the system being analyzed.
        num_nodes (int): Number of nodes in the system.
        n_atoms (int) : Number of atoms in atom groups represented by system nodes.
            Usually hydrogen atoms are not included in contact detection,
            and are not present in atom groups.
        atom_to_node (Any) : NumPy array that maps atoms in atom groups to their
            respective nodes.
        cutoff_dist (float): Distance cutoff used to capp distance calculations.
        node_group_indices_np (Any) : NumPy array with atom indices for all atoms
            in each node group.
        node_group_indices_np_aux (Any) : Auxiliary NumPy array with the indices of
            the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
        node_dists (Any) : Pre-allocated array to store cartesian distances.
        backend (str) : Controls how MDAnalysis will perform its distance
            calculations. Options are  `serial` and `openmp`. This option is
            ignored if the distance mode is not "all".
        dist_mode (int): Distance calculation method. Options are 0
            (for mode "all") and 1 (for mode "capped").
        verbose (int): Controls informational output.

    """

    assert isinstance(num_nodes, int)
    assert isinstance(n_atoms, int)
    assert isinstance(cutoff_dist, float)
    assert isinstance(backend, str)
    assert isinstance(verbose, int)

    assert dist_mode in [MODE_ALL, MODE_CAPPED]

    if verbose > 1:
        msg_str = "There are {} nodes and {} atoms in this system."
        print(msg_str.format(num_nodes, n_atoms))

        n_elements = int(n_atoms * (n_atoms - 1) / 2)
        print("creating array with {} elements...".format(n_elements))
        start = timer()

    if dist_mode == MODE_ALL:

        tmp_dists: npt.NDArray[np.float64] = \
            np.zeros(int(n_atoms * (n_atoms - 1) / 2), dtype=np.float64)

        if verbose > 1:
            end = timer()
            print("Time for matrix:", timedelta(seconds=end-start))

            print("running self_distance_array...")
            start = timer()

        # serial vs OpenMP
        mdadist.self_distance_array(selection.positions,
                                    result=tmp_dists, backend=backend)

        if verbose > 1:
            end = timer()
            print("Time for contact calculation:", timedelta(seconds=end-start))

    elif dist_mode == MODE_CAPPED:

        tmp_dists = np.full(int(n_atoms * (n_atoms - 1) / 2), cutoff_dist * 2,
                            dtype=float)

        if verbose > 1:
            end = timer()
            print("Time for matrix:", timedelta(seconds=end-start))

            print("running self_capped_distance...")
            start = timer()

        # method options are: 'bruteforce' 'nsgrid' 'pkdtree'
        pairs, distances = mdalibdist.self_capped_distance(selection.positions,
                                                           max_cutoff=cutoff_dist,
                                                           min_cutoff=None,
                                                           box=None,
                                                           method='pkdtree',
                                                           return_distances=True)

        if verbose > 1:
            end = timer()
            print("Time for contact calculation:", timedelta(seconds=end-start))

            print("Found {} pairs and {} distances".format(len(pairs), len(distances)))

            print("loading distances in array...")
            start = timer()
            if verbose > 1:
                start_loop = timer()

        for k in range(len(pairs)):
            i, j = pairs[k]

            if verbose > 1:
                if not k % 1000:
                    print("Loaded {} distances.".format(k))
                    elapsed_time = timedelta(seconds=timer() - start_loop)
                    print("Time for {} distances: {}".format(k, elapsed_time))
                    start_loop = timer()

            # Go from 2D node indices to 1D (numNodes*(numNodes-1)/2) indices:
            ijLI = get_lin_index_numba(i, j, n_atoms)
            tmp_dists[ijLI] = distances[k]

        if verbose > 1:
            end = timer()
            print("Time for loading distances:", timedelta(seconds=end-start))

    if verbose > 1:
        print("running atm_to_node_dist...")
        start = timer()

    # Translate atoms distances in minimum node distance.
    atm_to_node_dist(num_nodes,
                     n_atoms,
                     tmp_dists,
                     atom_to_node,
                     node_group_indices_np,
                     node_group_indices_np_aux,
                     node_dists)

    if verbose > 1:
        end = timer()
        print("Time for atm_to_node_dist:", timedelta(seconds=end-start))


@jit('void(i8, i8, f8, f8[:], f8[:], i8[:,:], i8[:], i8[:], i8[:])',
     nopython=True)
def calc_contact_c(num_nodes: int,
                   n_atoms: int,
                   cutoff_dist: float,
                   tmp_dists: npt.NDArray[np.float64],
                   tmp_dists_atms: npt.NDArray[np.float64],
                   contact_mat: npt.NDArray[np.int64],
                   atom_to_node: npt.NDArray[np.int64],
                   node_group_indices_np: npt.NDArray[np.int64],
                   node_group_indices_np_aux: npt.NDArray[np.int64]) \
        -> None:  # pragma: no cover
    """Translates MDAnalysis distance calculation to node contact matrix.

    This function is JIT compiled with Numba to optimize the search for nodes
    in contact.
    It relies on the results of MDAnalysis' `self_distance_array` calculation,
    stored in a 1D NumPy array of shape (n*(n-1)/2,), which acts as an unwrapped
    triangular matrix.

    In this function, the distances between all atoms in an atom groups of all
    pairs of nodes are verified to check if any pair of atoms were closer than
    a cutoff distance. This is done for all pairs of nodes in the system, and
    all frames in the trajectory. The pre-allocated contact matrix passed as an
    argument to this function is used to store the number of frames where each
    pair of nodes had at least one contact.

    Args:
        num_nodes (int): Number of nodes in the system.
        n_atoms (int) : Number of atoms in atom groups represented by system nodes.
            Usually hydrogen atoms are not included in contact detection, and
            are not present in atom groups.
        cutoff_dist (float) : Distance at which atoms are no longer considered
            'in contact'.
        tmp_dists (Any) : Temporary pre-allocated NumPy array with atom distances.
            This is the result of MDAnalysis `self_distance_array` calculation.
        tmp_dists_atms (Any) : Temporary pre-allocated NumPy array to store the
            shortest distance between atoms in different nodes.
        contact_mat (Any) : Pre-allocated NumPy matrix where node contacts will
            be stored.
        atom_to_node (Any) : NumPy array that maps atoms in atom groups to their
            respective nodes.
        node_group_indices_np (Any) : NumPy array with atom indices for all atoms
            in each node group.
        node_group_indices_np_aux (Any) : Auxiliary NumPy array with the indices of
            the first atom in each atom group, as listed in `nodeGroupIndicesNP`.

    """

    # We iterate until we have only one node left
    for i in range(num_nodes - 1):

        # Initializes the array for this node.
        tmp_dists_atms.fill(cutoff_dist * 2)

        # index of first atom in node "i"
        node_atm_indx = node_group_indices_np_aux[i]
        # index of first atom in node "i+1"
        node_atm_indx_next = node_group_indices_np_aux[i + 1]

        # Gets first atom in next node
        next_i_first = node_group_indices_np[node_atm_indx_next]

        # Per node:  Iterate over atoms from node i
        for node_i_k in node_group_indices_np[node_atm_indx:node_atm_indx_next]:

            # Go from 2D indices to 1D (n*(n-1)/2) indices:
            j1 = get_lin_index_numba(node_i_k, next_i_first, n_atoms)
            jend = j1 + (n_atoms - next_i_first)

            # Gets the shortest distance between atoms in different nodes
            tmp_dists_atms[next_i_first:] = np.where(
                tmp_dists[j1: jend] < tmp_dists_atms[next_i_first:],
                tmp_dists[j1: jend],
                tmp_dists_atms[next_i_first:])

        # Adds one to the contact to indicate that this frame had a contact.
        indices = np.unique(atom_to_node[np.where(tmp_dists_atms < cutoff_dist)[0]])
        for index in indices:
            contact_mat[i, index] += 1


@jit(nopython=True, parallel=True)
def _place_distances(tmp_dists: npt.NDArray[np.float64],
                     n_atoms: int,
                     pairs: Any,
                     distances: Any) -> None:  # pragma: no cover
    """Result processing for self_capped_distance

    This function converts the pair-distance results from the MDAnalysis function
    self_capped_distance and places it in a NumPy array for further processing.

    Args:
        tmp_dists (Any) : Temporary pre-allocated NumPy array for atom distances.
        n_atoms (int) : Number of atoms in atom groups represented by system nodes.
        pairs (Any) : Results from self_capped_distance, listing pairs of nodes for
            which distances were calculated.
        distances (Any): The mathing distances calculated by self_capped_distance
            for the pairs listed in `pairs`.

    Returns:
        None

    """
    for k in prange(len(pairs)):
        i, j = pairs[k]
        # Go from 2D node indices to 1D (n_atoms*(n_atoms-1)/2) indices:
        ijLI = get_lin_index_numba(i, j, n_atoms)
        tmp_dists[ijLI] = distances[k]


# High memory usage (nAtoms*(nAtoms-1)/2), calcs all atom distances at once.
def get_contacts_c(selection: mda.AtomGroup,
                   num_nodes: int,
                   n_atoms: int,
                   cutoff_dist: float,
                   tmp_dists: npt.NDArray[np.float64],
                   tmp_dists_atms: npt.NDArray[np.float64],
                   contact_mat: npt.NDArray[np.int64],
                   atom_to_node: npt.NDArray[np.int64],
                   node_group_indices_np: npt.NDArray[np.int64],
                   node_group_indices_np_aux: npt.NDArray[np.int64],
                   dist_mode: int = MODE_ALL) -> None:
    """Executes MDAnalysis atom distance calculation and node contact detection.

    This function is JIT compiled with Numba as a wrapper for two optimized distance
    calculation and contact determination calls. The first is MDAnalysis'
    `self_distance_array`. The second is the internal :py:func:`calc_contact_c`.
    All results are stored in pre-allocated NumPy arrays.

    Args:
        selection (MDAnalysis.AtomGroup) : Atom selection for the system being analyzed.
        num_nodes (int): Number of nodes in the system.
        n_atoms (int) : Number of atoms in atom groups represented by system nodes.
            Usually hydrogen atoms are not included in contact detection, and
            are not present in atom groups.
        cutoff_dist (float) : Distance at which atoms are no longer
            considered 'in contact'.
        tmp_dists (Any) : Temporary pre-allocated NumPy array with atom distances.
            This is the result of MDAnalysis `self_distance_array` calculation.
        tmp_dists_atms (Any) : Temporary pre-allocated NumPy array to store the
            shortest distance between atoms in different nodes.
        contact_mat (Any) : Pre-allocated NumPy matrix where node contacts will
            be stored.
        atom_to_node (Any) : NumPy array that maps atoms in atom groups to their
            respective nodes.
        node_group_indices_np (Any) : NumPy array with atom indices for all atoms
            in each node group.
        node_group_indices_np_aux (Any) : Auxiliary NumPy array with the indices of
            the first atom in each atom group, as listed in `nodeGroupIndicesNP`.
        dist_mode (int): Method for distance calculation in MDAnalysis (all or capped).

    """

    if dist_mode == MODE_ALL:
        # serial vs OpenMP
        mdadist.self_distance_array(selection.positions,
                                    result=tmp_dists,
                                    backend='openmp')

    if dist_mode == MODE_CAPPED:
        # method options are: 'bruteforce' 'nsgrid' 'pkdtree'
        pairs, distances = mdalibdist.self_capped_distance(
            reference=selection.positions,
            max_cutoff=cutoff_dist,
            min_cutoff=None,
            box=None,
            method='pkdtree',
            return_distances=True)

        _place_distances(tmp_dists, n_atoms, pairs, distances)

    calc_contact_c(num_nodes, n_atoms, cutoff_dist, tmp_dists,
                   tmp_dists_atms, contact_mat, atom_to_node,
                   node_group_indices_np, node_group_indices_np_aux)
