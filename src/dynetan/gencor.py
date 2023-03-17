#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr

import MDAnalysis

import numpy as np
import numpy.typing as npt

from multiprocessing import Queue

# For generalized correlations
from numba import jit
from numba import prange

##################################################
##################################################

# Auxiliary functions for calculation of correlation coefficients.


@jit('void(f8[:,:,:], i8, i8)', nopython=True, parallel=True)
def stand_vars_c(traj: npt.NDArray[np.float64],
                 num_nodes: int,
                 num_dims: int) -> None:  # pragma: no cover
    """Standardize variables in trajectory data.

    This function prepares the trajectory for the estimation of mutual
    information coefficients. This calculation assumes that the input trajectory
    of random variables (in this case, positions of atoms) is provided in a
    variable-dimension-step format (or "atom by x/y/z by frame" for molecular
    dynamics data).

    .. note:: Please refer to :py:func:`prep_mi_c` for details about the data
        conversion process. Please refer to :py:func:`calc_mir_numba_2var` for details
        about the calculation of mutual information coefficients.

    Args:
        traj (Any): NumPy array with trajectory information.
        num_nodes (int): Number of nodes in the network.
        num_dims (int): Number of dimensions in trajectory data
            (usually 3 dimensions, for X,Y,Z coordinates).

    """

    # Standardize variables
    for atm in prange(num_nodes):
        for dim in prange(num_dims):
            # Normalize each dimension.
            traj[atm, dim, :] = (traj[atm, dim, :] -
                                 traj[atm, dim, :].mean()) / traj[atm, dim, :].std()

            # Offset all data by minimum normalized value.
            traj[atm, dim, :] -= traj[atm, dim, :].min()


def prep_mi_c(universe: MDAnalysis.Universe,
              traj: npt.NDArray[np.float64],
              beg: int,
              end: int,
              num_nodes: int,
              num_dims: int) -> None:
    """Standardize variables in trajectory data.

    This function stores the trajectory data in a new format to accelerate the
    estimation of mutual information coefficients. The estimation of mutual
    information coefficients assumes that the input trajectory of random variables
    (in this case, positions of atoms) is provided in a variable-dimension-step
    format (or "atom by x/y/z by frame" for molecular dynamics data). However,
    the standard MDAnalysis trajectory format if "frame by atom by x/y/z".

    This function also calls a dedicated function to standardize the atom
    position data, also necessary for mutual information estimation.

    .. note:: Please refer to :py:func:`stand_vars_c` for details about the data
        standardization process. Please refer to :py:func:`calc_mir_numba_2var` for
        details about the calculation of mutual information coefficients.

    Args:
        universe (Any): MDAnalysis universe object containing all
            trajectory information.
        traj (Any): NumPy array where trajectory information will be stored.
        beg (int): Initial trajectory frame to be used for analysis.
        end (int): Final trajectory frame to be used for analysis.
        num_nodes (int): Number of nodes in the network.
        num_dims (int): Number of dimensions in trajectory data
            (usually 3 dimensions, for X,Y,Z coordinates).

    """

    # Copy trajectory
    for frame_index, ts in enumerate(universe.trajectory[beg:end]):

        for atm_index in range(num_nodes):
            for dim in range(num_dims):
                traj[atm_index, dim, frame_index] = ts.positions[atm_index, dim]

    stand_vars_c(traj, num_nodes, num_dims)


@jit('f8(f8[:,:,:], i4, i8, i4, f8[:], f8[:])', nopython=True)
def calc_mir_numba_2var(traj: npt.NDArray[np.float64],
                        num_frames: int,
                        num_dims: int,
                        k_neighb: npt.NDArray[np.float64],
                        psi: npt.NDArray[np.float64],
                        phi: npt.NDArray[np.float64]) -> float:  # pragma: no cover
    """Calculate mutual information coefficients.

    This function estimates the mutual information coefficient based on
    `Kraskov et. al. (2004) <https://doi.org/10.1103/PhysRevE.69.066138>`_,
    using the rectangle method. This implementation is hardcoded for 2 variables,
    to maximize efficiency, and is Just-In-Time (JIT) compiled using `Numba`_.

    .. _Numba: https://numba.pydata.org/

    This calculation assumes that the input trajectory of two random variables
    (in this case, positions of two atoms) is provided in a
    variable-dimension-step format (or "atom by x/y/z by frame" for molecular
    dynamics data). It also assumes that all trajectory data has been
    standardized (see :py:func:`stand_vars_c`).

    Args:
        traj (Any): NumPy array with trajectory information.
        num_frames (int): Number of trajectory frames in the current window.
        num_dims (int): Number of dimensions in trajectory data
            (usually 3 dimensions, for X,Y,Z coordinates).
        k_neighb (int): Parameter used for mutual information estimation.
        psi (numpy.ndarray): Pre-calculated parameter used for mutual
            information estimation.
        phi (numpy.ndarray): Pre-calculated parameter used for mutual
            information estimation.

    """

    dxy = 0.0

    diffX: npt.NDArray[np.float64] = np.zeros(num_frames, dtype=np.float64)
    diffY: npt.NDArray[np.float64] = np.zeros(num_frames, dtype=np.float64)
    tmpDiff: npt.NDArray[np.float64] = np.zeros(num_frames, dtype=np.float64)
    sortIndx: npt.NDArray[np.int64] = np.zeros(num_frames, dtype=np.int64)

    for step in range(num_frames):
        diffX.fill(0)
        diffY.fill(0)
        tmpDiff.fill(0)
        sortIndx.fill(0)

        for d in range(num_dims):

            tmpDiff = np.abs(traj[0, d, :] - traj[0, d, step])

            diffX = np.where(diffX > tmpDiff, diffX, tmpDiff)

            tmpDiff = np.abs(traj[1, d, :] - traj[1, d, step])

            diffY = np.where(diffY > tmpDiff, diffY, tmpDiff)

        # Create an array with indices of sorted distance arrays
        sortIndx = np.argsort(np.where(diffX > diffY, diffX, diffY))

        epsx = 0
        epsy = 0

        # Get the maximum distance in each dimension, for each variable,
        #  among k nearest neighbors.
        # We add one to the count to include the k-th neighbour, as the first index
        #  in the list it the frame itself.
        for kindx in range(1, k_neighb + 1):

            for d in range(num_dims):
                # For variable "i"
                dist = np.abs(traj[0, d, step] - traj[0, d, sortIndx[kindx]])

                if epsx < dist:
                    epsx = dist

                # For variable "j"
                dist = np.abs(traj[1, d, step] - traj[1, d, sortIndx[kindx]])

                if epsy < dist:
                    epsy = dist

        # Count the number of frames in which the point is within "eps-" distance from
        #   the position in frame "step". Subtract one so not to count the frame itself.
        nx = len(np.nonzero(diffX <= epsx)[0]) - 1
        ny = len(np.nonzero(diffY <= epsy)[0]) - 1

        dxy += psi[nx] + psi[ny]

    dxy /= num_frames

    # Mutual Information R
    return psi[num_frames] + phi[k_neighb] - dxy


def calc_cor_proc(traj: npt.NDArray[np.float64],
                  win_len: int,
                  psi: npt.NDArray[np.float64],
                  phi: npt.NDArray[np.float64],
                  num_dims: int,
                  k_neighb: npt.NDArray[np.float64],
                  in_queue: Queue,
                  out_queue: Queue) -> None:
    """Process for parallel calculation of generalized correlation coefficients.

    This function serves as a wrapper and manager for the calculation of
    generalized correlation coefficients. It uses Python's `multiprocessing`
    module to launch `Processes` for parallel execution, where each process uses
    two `multiprocessing` queues to manage job acquisition and saving results.
    For each calculation, the job acquisition queue passes a trajectories of a
    pair of atoms, while the output queue stores the generalized correlation
    coefficient. The generalized correlation coefficient is calculated using a
    mutual information coefficient which is estimated using an optimized function.

    .. note:: Please refer to :py:func:`calc_mir_numba_2var` for details about the
        calculation of mutual information coefficients.

    Args:
        traj (Any): NumPy array with trajectory information.
        win_len (int): Number of trajectory frames in the current window.
        psi (Any): Pre-calculated parameter used for mutual information estimation.
        phi (Any): Pre-calculated parameter used for mutual information estimation.
        num_dims (int): Number of dimensions in trajectory data
            (usually 3 dimensions, for X,Y,Z coordinates).
        k_neighb (int): Parameter used for mutual information estimation.
        in_queue (Any) : Multiprocessing queue object for acquiring jobs.
        out_queue (Any) : Multiprocessing queue object for placing results.

    """

    import queue  # So that we can catch the exceptions for empty queues

    # While we still have elements to process, get a new pair of nodes.
    # We verify if we run out of elements by catching a termination flag
    # in the queue. There is at least one termination flag per initiated process.
    while True:

        try:
            atmList = in_queue.get(block=True, timeout=0.01)
        except queue.Empty:  # pragma: no cover
            continue
            # If we need to wait for longer for a new item,
            # just continue the loop
        else:
            # The termination flag is an empty list
            if len(atmList) == 0:
                break

            # Calls the Numba-compiled function.
            mir = calc_mir_numba_2var(traj[atmList, :, :],
                                      win_len,
                                      num_dims,
                                      k_neighb,
                                      psi,
                                      phi)

            out_queue.put((atmList, mir))


@jit(nopython=True, nogil=True)
def mir_to_corr(mir: float, num_dims: int = 3) -> float:  # pragma: no cover
    """Transforms Mutual Information R into Generalized Correlation Coefficient

    Returns:
        Generalized Correlation Coefficient (float)
    """

    # Assures that the Mutual Information estimate is not lower than zero.
    corr = max(0.0, mir)

    # Determine generalized correlation coefficient from the Mutual Information
    corr = np.sqrt(1 - np.exp(-corr * (2.0 / num_dims)))

    return corr
