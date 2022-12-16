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
    
def formatNodeGroups(atmGrp, nodeAtmStrL=["CA"], grpAtmStrL=None):
    '''Format code to facilitate the definition of node groups.

    This convenience function helps with the definition of node groups.
    It will produce formated python code that the user can copy directly
    into a definition of atom groups.

    If the entire residue is to be represented by a single node, then `grpAtmStrL`
    does not need to be defined. However, if more than one node is defined in `nodeAtmStrL`,
    then the same number of lists need to be added to `grpAtmStrL` to define each node group.

    Args:
        atmGrp (MDanalysis atom group object): The atom group object containing one residue.
        nodeAtmStrL (list): A list of strings defining atoms that will represent nodes in network analysis.
        grpAtmStrL (list): A list of lists containing atoms that belong to each node group.

    Returns:
        ---

    '''

    if not (isinstance(nodeAtmStrL,list)):
        print("ERROR: Expected a list argument for nodeAtmStr, but received: {}.".format(type(nodeAtmStrL).__name__))
        return

    # We use this to check if the input is making sense, and to get the residue name.
    if len(atmGrp.residues) != 1:
        print("ERROR: Expected 1 residue in atom group, but received {}.".format(len(atmGrp.residues)))
        return

    atmNames = list(atmGrp.names)
    resName  = atmGrp.resnames[0]

    if not (set(nodeAtmStrL).issubset(set(atmNames))):
        errorSet = set(nodeAtmStrL) - set(atmNames)
        print("ERROR: The following node atoms are present in the residue: {}".format(errorSet))
        return

    print("""
        You can copy and paste the following lines into your notebook to define
        the node group(s) for your new residue.
        """)

    print("usrNodeGroups[\"{}\"] = {{}}".format(resName) )

    if len(nodeAtmStrL) == 1:
        print("usrNodeGroups[\"{}\"][\"{}\"] = {}".format(
            resName, nodeAtmStrL[0], set(list(atmGrp.names))) )

    else:

        if not grpAtmStrL:
            print("ERROR: The argument `grpAtmStrL` is not defined.")
            return

        if len(grpAtmStrL) != len(nodeAtmStrL):
            print("ERROR: Expected {} lists of atoms in `grpAtmStrL`, but received {}.".format(len(nodeAtmStrL), len(grpAtmStrL)))
            return

        for nodeAtmStr, grpAtmStr in zip(nodeAtmStrL,grpAtmStrL):
            print("usrNodeGroups[\"{}\"][\"{}\"] = {}".format(
                resName, nodeAtmStr, set(grpAtmStr) ) )

def showNodeGrps(w, atmGrp, usrNodeGroups, nodeAtmSel=""):
    '''Labels atoms in an NGLview instance to visualize node groups.

    This convenience function helps with the definition of node groups.
    It will label atoms and nodes in a structure to help visualize the selection
    of atoms and nodes.

    Args:
        w (nglview object): The initialized NGLview object.
        atmGrp (MDanalysis atom group object): The atom group object containing one residue.
        usrNodeGroups (dict): A dictionary of dictionaries with node groups for a given residue.
        nodeAtmSel (str): A string selecting a node atom so that only atoms in that group are labeled.

    Returns:
        ---

    '''

    selectedAtoms = set()
    for resName,nodeGrp in usrNodeGroups.items():
        for nodeAtm,grpAtms in nodeGrp.items():
            if (nodeAtmSel!="") and (nodeAtmSel!=nodeAtm):
                continue
            selTxt = ["." + atmStr for atmStr in grpAtms if atmStr != nodeAtm ]
            selTxt = " ".join(selTxt)

            w.add_representation(repr_type="label", selection=selTxt, labelType="atomname", color="black")
            w.add_representation(repr_type="label", selection="."+nodeAtm, labelType="atomname", color="green")

            selectedAtoms.update(set(grpAtms))

    # Create a set of atoms not selected by any node group.
    unSelectedAtoms = set(atmGrp.names) - selectedAtoms

    if len(unSelectedAtoms) and (nodeAtmSel==""):
        selTxt = ["." + atmStr for atmStr in unSelectedAtoms ]
        selTxt = " ".join(selTxt)

        w.add_representation(repr_type="label", selection=selTxt, labelType="atomname", color="red")
