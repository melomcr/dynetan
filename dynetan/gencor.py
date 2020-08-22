#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


import numpy as np
import cython

# For generalized correlations
from numba import jit

##################################################
##################################################

## Auxiliary functions for calculation of correlation coefficients.

@cython.cfunc
@cython.returns(cython.void)
@cython.locals(numNodes=cython.int, numDims=cython.int, traj="np.ndarray[np.float_t, ndim=3]")
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def standVarsC(traj, numNodes, numDims):
    '''Standardize variables in trajectory data.
    
    This function prepares the trajectory for the estimation of mutual information coefficients.
    This calculation assumes that the input trajectory of random variables (in this case, positions of atoms) is provided in a variable-dimension-step format (or "atom by x/y/z by frame" for molecular dynamics data). 
        
    .. note:: Please refer to :py:func:`prepMIc` for details about the data conversion process. Please refer to :py:func:`calcMIRnumba2var` for details about the calculation of mutual information coefficients.
    
    Args:
        traj (obj): NumPy array with trajectory information.
        numNodes (int): Number of nodes in the network.
        numDims (int): Number of dimensions in trajectory data (usually 3 dimensions, for X,Y,Z coordinates).
    
    '''
    
    # Standardize variables
    for atm in range(numNodes):
        for dim in range(numDims):
            # Noromalize each dimension.
            traj[atm, dim, :] = (traj[atm, dim, :] - traj[atm, dim, :].mean())/ traj[atm, dim, :].std()

            # Offset all data by minimum normalilzed value.
            traj[atm, dim, :] -= traj[atm, dim, :].min()

@cython.cfunc
@cython.returns(cython.void)
@cython.locals(traj="np.ndarray[np.float_t, ndim=3]",
               beg=cython.int, 
               end=cython.int, 
               numNodes=cython.int, 
               numDims=cython.int)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def prepMIc(universe, traj, beg, end, numNodes, numDims):
    '''Standardize variables in trajectory data.
    
    This function stores the trajectory data in a new format to accelerate the estimation of mutual information coefficients.
    The estimation of mutual information coefficients assumes that the input trajectory of random variables (in this case, positions of atoms) is provided in a variable-dimension-step format (or "atom by x/y/z by frame" for molecular dynamics data). However, the standard MDAnalysis trajectory format if "frame by atom by x/y/z".
    
    This function also calls a dedicated function to standardize the atom position data, also necessary for mutual information estimation.
        
    .. note:: Please refer to :py:func:`standVarsC` for details about the data standardization process. Please refer to :py:func:`calcMIRnumba2var` for details about the calculation of mutual information coefficients.
    
    Args:
        universe (obj): MDAnalysis universe object containing all trajectory information.
        traj (obj): NumPy array where trajectory information will be stored.
        beg (int): Initial trajectory frame to be used for analysis.
        end (int): Final trajectory frame to be used for analysis.
        numNodes (int): Number of nodes in the network.
        numDims (int): Number of dimensions in trajectory data (usually 3 dimensions, for X,Y,Z coordinates).
    
    '''
    
    # Copy trajectory
    for frameIndx, ts in enumerate(universe.trajectory[beg:end]):
    
        for atmIndx in range(numNodes):
            for dim in range(numDims):
                traj[atmIndx, dim, frameIndx] = ts.positions[atmIndx,dim]
    
    standVarsC(traj, numNodes, numDims)
 
#
# This takes ~4.52 ms ± 43.1 µs for 250 frames
#
@jit('f8(f8[:,:,:], i4, i8, i4, f8[:], f8[:])', nopython=True)
def calcMIRnumba2var(traj, numFrames, numDims, kNeighb, psi, phi):
    '''Calculate mutual information coefficients.
    
    This function estimates the mutual information coefficient based on `Kraskov et. al. (2004) <https://doi.org/10.1103/PhysRevE.69.066138>`_, using the rectangle method. This implementation is hardcoded for 2 variables, to maximize efficiency, and is Just-In-Time (JIT) compiled using `Numba`_.
    
    .. _Numba: https://numba.pydata.org/

    This calculation assumes that the input trajectory of two random variables (in this case, positions of two atoms) is provided in a variable-dimension-step format (or "atom by x/y/z by frame" for molecular dynamics data). It also assumes that all trajectory data has been standardized (see :py:func:`standVarsC`).
    
    Args:
        traj (obj): NumPy array with trajectory information.
        numFrames (int): Number of trajectory frames in the current window.
        numDims (int): Number of dimensions in trajectory data (usually 3 dimensions, for X,Y,Z coordinates).
        kNeighb (int): Parameter used for mutual information estimation.
        psi (float): Pre-calculated parameter used for mutual information estimation.
        phi (float): Pre-calculated parameter used for mutual information estimation.
    
    '''
    
    dxy = 0
    
    diffX = np.zeros(numFrames, dtype=np.float64)
    diffY = np.zeros(numFrames, dtype=np.float64)
    tmpDiff = np.zeros(numFrames, dtype=np.float64)
    sortIndx = np.zeros(numFrames, dtype=np.int64)
    
    for step in range(numFrames):
        diffX.fill(0)
        diffY.fill(0)
        tmpDiff.fill(0)
        sortIndx.fill(0)

        for d in range(numDims):
            
            tmpDiff = np.abs( traj[0,d,:] - traj[0,d,step] )
            
            diffX = np.where( diffX > tmpDiff, diffX, tmpDiff )
            
            tmpDiff = np.abs( traj[1,d,:] - traj[1,d,step] )
            
            diffY = np.where( diffY > tmpDiff, diffY, tmpDiff )
        
        # Create an array with indices of sorted distance arrays
        sortIndx = np.argsort( np.where(diffX > diffY, diffX, diffY))
        
        epsx = 0
        epsy = 0
        
        # Get the maximum diatance in each dimention, for each variable,
        #  among k nearest neighbors.
        # We add one to the count to include the k-th neighbour, as the first index
        #  in the list it the frame itself.
        for kindx in range(1, kNeighb+1):
            
            for d in range(numDims):
                # For variable "i"
                dist = np.abs( traj[0,d,step] - traj[0, d, sortIndx[kindx]] )
                
                if epsx < dist :
                    epsx = dist

                # For variable "j"
                dist = np.abs( traj[1,d,step] - traj[1, d, sortIndx[kindx]] )
                
                if epsy < dist :
                    epsy = dist
        
        # Count the number of frames in which the point is within "eps-" distance from
        #   the position in frame "step". Subtract one so not to count the frame itself.
        nx = len(np.nonzero( diffX <= epsx )[0]) -1
        ny = len(np.nonzero( diffY <= epsy )[0]) -1
        
        dxy += psi[nx] + psi[ny];
    
    dxy /= numFrames
    
    # Mutual Information R
    return psi[numFrames] + phi[kNeighb] - dxy

def calcCorProc(traj, winLen, psi, phi, numDims, kNeighb, inQueue, outQueue):
    '''Process for parallel calculation of generalized correlation coefficients.
    
    This function serves as a wrapper and manager for the calculation of generalized correlation coefficients. It uses Python's `multiprocessing` module to launch `Processes` for parallel execution, where each process uses two `multiprocessing` queues to manage job acquisition and saving results. 
    For each calculation, the job acquisition queue passes a trajectories of a pair of atoms, while the output queue stores the generalized correlation coefficient. 
    The generalized correlation coefficient is claculated using a mutual information coefficient which is estimated using an optimized function.
    
    .. note:: Please refer to :py:func:`calcMIRnumba2var` for details about the calculation of mutual information coefficients.
    
    Args:
        traj (obj): NumPy array with trajectory information.
        winLen (int): Number of trajectory frames in the current window.
        psi (float): Pre-calculated parameter used for mutual information estimation.
        phi (float): Pre-calculated parameter used for mutual information estimation.
        numDims (int): Number of dimensions in trajectory data (usually 3 dimensions, for X,Y,Z coordinates).
        kNeighb (int): Parameter used for mutual information estimation.
        inQueue (obj) : Multiprocessing queue object for acquiring jobs.
        outQueue (obj) : Multiprocessing queue object for placing results.
        
    '''
    
    # While we still hape elements to process, get a new pair of nodes. 
    while not inQueue.empty():
        
        atmList = inQueue.get()
        
        # Calls the Numba-compiled function.
        corr = calcMIRnumba2var(traj[atmList, :, :], winLen, numDims, kNeighb, psi, phi)
        
        # Assures that the Mutual Information estimate is not lower than zero.
        corr = max(0, corr)
        
        # Determine generalized correlation coeff from the Mutual Information
        corr = np.sqrt(1-np.exp(-corr*(2.0/3)));
            
        #return (atmList, corr)
        outQueue.put( (atmList, corr) )




