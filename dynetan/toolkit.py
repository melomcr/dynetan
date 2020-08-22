#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


# For trajectory analysis
import MDAnalysis as mda

def diagnostic():
    '''Diagnostic for parallelization of MDAnalysis.
    
    Convenience function to detect if the current MDAnalysis
    installation supports OpenMP.
    
    '''
    
    return mda.lib.distances.USED_OPENMP

## Progress Bar with time estimate
## Based on https://github.com/alexanderkuk/log-progress
def log_progress(sequence: list, every=None, size=None, name='Items', userProgress=None):
    '''Creates a progress bar in jupyter notebooks.
    
    Automatically detects the size of a list and estimates the best step
    size for progress bar updates. This function also automatically estimates
    the total time to completion of the iterations, updating the estimate using 
    the time that every step takes.
    
    If the sequence argument is an iterator, the total number of elements cannot 
    determined. In this case, the user must define the `every` parameter to indicate
    the update frequency of the progress bar.
    
    If the progress bar is used in a nested loop, passing a list to the `userProgress` 
    argument will force the re-utilization of `ipywidgets` objects, preventing the 
    creation of a new progress bar at every iteration of the inner loop.
    
    This progress bar was based on https://github.com/alexanderkuk/log-progress.
    
    Args:
        sequence : An iterable object.
        every (int): The update frequency.
        size (int): The number of elements in the sequence.
        name (str): The name of the progress bar.
        userProgress (list): List for creation of nested progress bars.
    
    '''
    from ipywidgets import IntProgress, HTML, HBox, Label
    from IPython.display import display
    from numpy import mean as npmean
    from collections import deque
    from math import floor
    from datetime import datetime
    from string import Template
    
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = floor(float(size)*0.005)     # every 0.5%, minimum is 1
    else:
        assert every is not None, 'sequence is iterator, set every'
    
    # For elapsed time
    initTime = datetime.now()
    totTime = "?"
    labTempl = Template(" (~ $min total time (min) ; $ell minutes elapsed)")
    
    # If provided, we use the objects already created.
    # If not provided, we create from scratch.
    if userProgress is None or userProgress == []:
        
        progress = IntProgress(min=0, max=1, value=1)

        label = HTML()
        labelTime = Label("")

        box = HBox(children=[label, progress, labelTime])
        
        if userProgress == []:
            userProgress.append(box)
        display(box)
    else:
        box = userProgress[0]
    
    if is_iterator:
        #progress = IntProgress(min=0, max=1, value=1)
        box.children[1].min = 0
        box.children[1].max = 1
        box.children[1].value = 1
        box.children[1].bar_style = 'info'
    else:
        #progress = IntProgress(min=0, max=size, value=0)
        box.children[1].min = 0
        box.children[1].max = size
        box.children[1].value = 0

        # For remaining time estimation
        deltas = deque()
        lastTime = None
        meandelta = 0
    
    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    box.children[0].value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    box.children[1].value = index
                    box.children[0].value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
                
                    # Estimates remaining time with average delta per iteration
                    # Uses (at most) the last 30 iterations
                    if len(deltas) == 101:
                        deltas.popleft()
                    
                    if lastTime:
                        deltas.append( (datetime.now() - lastTime).total_seconds() )
                        meandelta = npmean(deltas)/60.0    # From seconds to minute
                        totTime = round(meandelta*size/float(every), 3)  # Mean iteration for all iterations
                    else:
                        totTime = "?"       # First iteration has no time
                    
                    lastTime = datetime.now()
                
                # All ellapsed time in minutes
                elapsed = round( (datetime.now() - initTime).total_seconds()/60.0, 3)

                box.children[2].value = labTempl.safe_substitute({"min":totTime,
                                                       "ell":elapsed})
                
            yield record
    except:
        box.children[1].bar_style = 'danger'
        raise
    else:
        box.children[1].bar_style = 'success'
        box.children[1].value = index
        box.children[0].value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def getNGLSelFromNode(nodeIndx, atomsel, atom=True):
    '''Creates an atom selection for NGLView.
    
    Retunrs an atom selection for a whole residue or single atom in the NGL syntax.
    
    Arguments:
        nodeIndx (int): 
            Index of network node.
        atomsel (obj): 
            MDAnalysis atom-selection object.
        atom (bool): 
            Determines if the selection should cover the entire residue, or just the representative atom.
    
    Returns:
        str: Text string with NGL-style atom selection.
    
    '''
    node = atomsel.atoms[nodeIndx]
    if atom:
        return " and ".join([str(node.resid), node.resname, "." + node.name])
    else:
        return " and ".join([str(node.resid), node.resname])

def getNodeFromSel(selection, atomsel, atomToNode):
    '''Gets the node index from an atom selection. 
    
    Returns one or more node indices when given an MDAnalysis atom selection string.
    
    Arguments:
        selection (str):
            MDAnalysis atom selection string.
        atomsel (obj):
            MDAnalysis atom-selection object.
        atomToNode (obj):
            Dynamic Network Analysis atom-to-node mapping object.
    
    Returns:
        np.array : List of node indices mapped to the provided atom selection.
    
    '''
    
    nodes = atomToNode[ atomsel.select_atoms(selection).ix_array ]
    
    # There may be atoms not assigned to nodes, e.g., if a whole nucleotide was
    #   kept but only one of its nodes had contacts.
    return nodes[ nodes >= 0 ]

def getSelFromNode(nodeIndx, atomsel, atom=False):
    '''Gets the MDanalysis selection string from a node index.
    
    Given a node index, this function builds an atom selection string in the 
    following format: resname and resid and segid [and name]
    
    Arguments:
        nodeIndx (int): 
            Index of network node.
        atomsel (obj): 
            MDAnalysis atom-selection object.
        atom (bool): 
            Determines if the selection should cover the entire residue, or just the representative atom.
    
    Returns:
        str: Text string with MDAnalysis-style atom selection.
        
    '''
    nodeIndx = int(nodeIndx)
    if nodeIndx < 0:
        raise
    resName = atomsel.atoms[nodeIndx].resname
    resID = str(atomsel.atoms[nodeIndx].resid)
    segID = atomsel.atoms[nodeIndx].segid
    atmName = atomsel.atoms[nodeIndx].name
    
    if atom:
        return "resname " + resName + " and resid " + resID + " and segid " + segID + " and name " + atmName
    else:
        return "resname " + resName + " and resid " + resID + " and segid " + segID


def getPath(src, trg, nodesAtmSel, preds, win=0):
    '''Gets connecting path between nodes.
    
    This function recovers the list of nodes that connect `src` and `trg` nodes.
    An internal sanity check is performed to see if both nodes belong to the same residue. 
    This may be the case in nucleic acids, for example, where two nodes are used to describe
    the entire residue.
    
    Args:
        src (int): Source node.
        trg (int): Target node.
        nodesAtmSel (obj): MDAnalysis atom-selection object.
        win (int): Selects the simulation window used to create optimal paths.
    
    Returns:
        np.array: A NumPy array with the list of nodes or an empty list in case 
        no optimal path could be found.
    
    '''

    import numpy as np

    src = int(src)
    trg = int(trg)

    if src == trg:
        return []

    if src not in preds[win].keys():
        return []

    if trg not in preds[win][src].keys():
        return []

    if getSelFromNode(src, nodesAtmSel) == getSelFromNode(trg, nodesAtmSel):
        return []

    path = [trg]

    while path[-1] != src:
        path.append(preds[win][src][path[-1]])

    return np.asarray(path)

def getLinIndexC( src, trgt, dim):
    '''Conversion from 2D matrix indices to 1D triangular.
    
    Converts from 2D matrix indices to 1D (n*(n-1)/2) unwrapped triangular matrix index.
    
    Args:
        src (int): Source node.
        trg (int): Target node.
        dim (int): Dimension of square matrix
    
    Returns:
        int: 1D index in unwrapped triangular matrix.
    
    '''
    
    #based on https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    k = (dim*(dim-1)/2) - (dim-src)*((dim-src)-1)/2 + trgt - src - 1.0
    return int(k)

def getCartDist(src,trgt, numNodes, nodeDists, distype=0):
    '''Get cartesian distance between nodes.
    
    Retreives the cartesian distance between atoms representing nodes `src` and `trgt`.
    The `distype` argument causes the function to return the mean distance (type 0: default), Standard Error of the Mean (SEM) (type 1), minimum distance (type 2), or maximum distance (type 3).
    
    Args:
        src (int): Source node.
        trg (int): Target node.
        numNodes (int): Total number of nodes in the system.
        distype (int): Type of cartesian distance output.
    
    Returns:
        float: 
            One of four types of measurements regarding the cartesian distance between nodes. See description above.
    
    '''
    
    if src == trgt:
        return 0
    elif trgt < src:
        # We need to access the list with smaller index on row, larger on the column
        src, trgt = trgt, src
    
    k = getLinIndexC(src, trgt, numNodes)
    
    return nodeDists[distype, k]
    
