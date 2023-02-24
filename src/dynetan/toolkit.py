#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr

# For trajectory analysis
import MDAnalysis as mda
import numpy
from typing import Any, Union


def diagnostic() -> bool:
    """Diagnostic for parallelization of MDAnalysis.

    Convenience function to detect if the current MDAnalysis
    installation supports OpenMP.

    """

    return mda.lib.distances.USED_OPENMP


def getNGLSelFromNode(nodeIndx: int, atomsel: mda.AtomGroup, atom: bool = True) -> str:
    """Creates an atom selection for NGLView.

    Returns an atom selection for a whole residue or single atom in the NGL
    syntax.

    Arguments:
        nodeIndx (int):
            Index of network node.
        atomsel (Any):
            MDAnalysis atom-selection object.
        atom (bool):
            Determines if the selection should cover the entire residue, or just
            the representative atom.

    Returns:
        str: Text string with NGL-style atom selection.

    """
    node = atomsel.atoms[nodeIndx]
    if atom:
        return " and ".join([str(node.resid), node.resname, "." + node.name])
    else:
        return " and ".join([str(node.resid), node.resname])


def getNodeFromSel(selection: str,
                   atmsel: mda.AtomGroup,
                   atm_to_node: numpy.ndarray) -> numpy.ndarray:
    """Gets the node index from an atom selection.

    Returns one or more node indices when given an MDAnalysis atom selection
    string.

    Arguments:
        selection (str):
            MDAnalysis atom selection string.
        atmsel (Any):
            MDAnalysis atom-selection object.
        atm_to_node (Any):
            Dynamic Network Analysis atom-to-node mapping object.

    Returns:
        numpy.ndarray : List of node indices mapped to the provided atom selection.

    """

    nodes = atm_to_node[atmsel.select_atoms(selection).ix_array]

    # There may be atoms not assigned to nodes, e.g., if a whole nucleotide was
    #   kept but only one of its nodes had contacts.
    return nodes[nodes >= 0]


def getSelFromNode(nodeIndx: int,
                   atomsel: mda.AtomGroup,
                   atom: bool = False) -> str:
    """Gets the MDanalysis selection string from a node index.

    Given a node index, this function builds an atom selection string in the
    following format: resname and resid and segid [and name]

    Arguments:
        nodeIndx (int):
            Index of network node.
        atomsel (Any):
            MDAnalysis atom-selection object.
        atom (bool):
            Determines if the selection should cover the entire residue, or
            just the representative atom.

    Returns:
        str: Text string with MDAnalysis-style atom selection.

    """

    assert isinstance(nodeIndx, (int, numpy.integer))
    assert nodeIndx >= 0

    resName = atomsel.atoms[nodeIndx].resname
    resID = str(atomsel.atoms[nodeIndx].resid)
    segID = atomsel.atoms[nodeIndx].segid
    atmName = atomsel.atoms[nodeIndx].name

    retStr = "resname " + resName + " and resid " + resID + " and segid " + segID

    if atom:
        return retStr + " and name " + atmName
    else:
        return retStr


def getPath(src: int,
            trg: int,
            nodesAtmSel: mda.AtomGroup,
            preds: dict,
            win: int = 0) -> numpy.ndarray:
    """Gets connecting path between nodes.

    This function recovers the list of nodes that connect `src` and `trg` nodes.
    An internal sanity check is performed to see if both nodes belong to the same
    residue. This may be the case in nucleic acids, for example, where two nodes
    are used to describe the entire residue.

    Args:
        src (int): Source node.
        trg (int): Target node.
        nodesAtmSel (Any): MDAnalysis atom-selection object.
        preds (dict): Predecessor data in dictionary format.
        win (int): Selects the simulation window used to create optimal paths.

    Returns:
        numpy.ndarray: A NumPy array with the list of nodes or an empty list in case
            no optimal path could be found.

    """

    assert isinstance(src, (int, numpy.integer))
    assert isinstance(trg, (int, numpy.integer))
    assert isinstance(win, int)

    assert src >= 0
    assert trg >= 0
    assert win >= 0

    if src == trg:
        return numpy.asarray([])

    if src not in preds[win].keys():
        return numpy.asarray([])

    if trg not in preds[win][src].keys():
        return numpy.asarray([])

    if getSelFromNode(src, nodesAtmSel) == getSelFromNode(trg, nodesAtmSel):
        return numpy.asarray([])

    path = [trg]

    while path[-1] != src:
        path.append(preds[win][src][path[-1]])

    return numpy.asarray(path)


def getLinIndexC(src: int, trgt: int, dim: int) -> int:
    """Conversion from 2D matrix indices to 1D triangular.

    Converts from 2D matrix indices to 1D (n*(n-1)/2) unwrapped triangular
    matrix index.

    Args:
        src (int): Source node.
        trgt (int): Target node.
        dim (int): Dimension of square matrix

    Returns:
        int: 1D index in unwrapped triangular matrix.

    """

    assert isinstance(src, (int, numpy.integer))
    assert isinstance(trgt, (int, numpy.integer))
    assert isinstance(dim, (int, numpy.integer))

    assert src >= 0
    assert trgt >= 0
    assert dim >= 0

    # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    k = (dim*(dim-1)/2) - (dim-src)*((dim-src)-1)/2 + trgt - src - 1.0
    return int(k)


def getCartDist(src: int,
                trgt: int,
                numNodes: int,
                nodeDists: numpy.ndarray,
                distype: int = 0) -> float:
    """Get cartesian distance between nodes.

    Retrieves the cartesian distance between atoms representing nodes `src` and
    `trgt`. The `distype` argument causes the function to return the mean
    distance (type 0: default), Standard Error of the Mean (SEM) (type 1),
    minimum distance (type 2), or maximum distance (type 3).

    Args:
        src (int): Source node.
        trgt (int): Target node.
        numNodes (int): Total number of nodes in the system.
        distype (int): Type of cartesian distance output.

    Returns:
        float:
            One of four types of measurements regarding the cartesian distance
            between nodes. See description above.

    """

    assert isinstance(src, (int, numpy.integer))
    assert isinstance(trgt, (int, numpy.integer))
    assert isinstance(numNodes, (int, numpy.integer))

    assert src >= 0
    assert trgt >= 0
    assert numNodes > 0

    assert nodeDists is not None, "Cartesian distances not yet calculated!"

    if src == trgt:
        return 0
    elif trgt < src:
        # We need to access the list with smaller index on row,
        #  larger on the column
        src, trgt = trgt, src

    k = getLinIndexC(src, trgt, numNodes)

    return nodeDists[distype, k]


def formatNodeGroups(atmGrp: mda.AtomGroup,
                     nodeAtmStrL: list,
                     grpAtmStrL: Union[list, None] = None) -> None:
    """Format code to facilitate the definition of node groups.

    This convenience function helps with the definition of node groups.
    It will produce formatted python code that the user can copy directly
    into a definition of atom groups.

    If the entire residue is to be represented by a single node, then
    `grpAtmStrL` does not need to be defined. However, if more than one node is
    defined in `nodeAtmStrL`, then the same number of lists need to be added to
    `grpAtmStrL` to define each node group.

    Args:
        atmGrp (Any): MDAnalysis atom group object with one residue.
        nodeAtmStrL (list): Strings defining atoms that will represent nodes.
        grpAtmStrL (list|None): Lists containing atoms that belong to each node group.
            If None, all atoms in the residue will be added to the same node group.
            This parameter can only be None when a single node is provided.

    Returns:
        ---

    """

    # Check if the input is making sense, and to get the residue name.
    assert isinstance(nodeAtmStrL, list), "Expected a list of node atom name(s)."

    assert len(atmGrp.residues) == 1, "Expected a single residue in atom group."

    atmNames = list(atmGrp.names)
    resName = atmGrp.resnames[0]

    if not (set(nodeAtmStrL).issubset(set(atmNames))):
        errorSet = set(nodeAtmStrL) - set(atmNames)
        errStr = "ERROR: The following node atoms are NOT present in the residue: {}"
        print(errStr.format(errorSet))
        raise

    print("""
        You can copy and paste the following lines into your notebook to define
        the node group(s) for your new residue.
        """)

    print("usrNodeGroups[\"{}\"] = {{}}".format(resName))

    if len(nodeAtmStrL) == 1:
        print("usrNodeGroups[\"{}\"][\"{}\"] = {}".format(
            resName, nodeAtmStrL[0], set(list(atmGrp.names))))

    else:

        if not grpAtmStrL:
            print("ERROR: The argument `grpAtmStrL` is not defined.")
            raise

        if len(grpAtmStrL) != len(nodeAtmStrL):
            errStr = "ERROR: Expected {} lists of atoms in `grpAtmStrL` but received {}."
            print(errStr.format(len(nodeAtmStrL), len(grpAtmStrL)))
            raise

        for nodeAtmStr, grpAtmStr in zip(nodeAtmStrL, grpAtmStrL):
            print("usrNodeGroups[\"{}\"][\"{}\"] = {}".format(
                resName, nodeAtmStr, set(grpAtmStr)))


def showNodeGroups(nv_view: Any,
                   atm_grp: mda.AtomGroup,
                   usr_node_groups: dict,
                   node_atm_sel: str = "") -> None:
    """Labels atoms in an NGLview instance to visualize node groups.

    This convenience function helps with the definition of node groups.
    It will label atoms and nodes in a structure to help visualize the selection
    of atoms and nodes.

    Args:
        nv_view (Any): The initialized NGLview object.
        atm_grp (Any): The MDanalysis atom group object containing
            one residue.
        usr_node_groups (dict): A dictionary of dictionaries with node groups for
            a given residue.
        node_atm_sel (str): A string selecting a node atom so that only atoms in
            that group are labeled.

    Returns:
        ---

    """

    selectedAtoms = set()
    for resName, nodeGrp in usr_node_groups.items():
        for nodeAtm, grpAtms in nodeGrp.items():
            if (node_atm_sel != "") and (node_atm_sel != nodeAtm):
                continue
            selTxtL = ["." + atmStr for atmStr in grpAtms if atmStr != nodeAtm]
            selTxt = " ".join(selTxtL)

            nv_view.add_representation(repr_type="label",
                                       selection=selTxt,
                                       labelType="atomname",
                                       color="black")
            nv_view.add_representation(repr_type="label",
                                       selection="."+nodeAtm,
                                       labelType="atomname",
                                       color="green")

            selectedAtoms.update(set(grpAtms))

    # Create a set of atoms not selected by any node group.
    unSelectedAtoms = set(atm_grp.names) - selectedAtoms

    if len(unSelectedAtoms) and (node_atm_sel == ""):
        selTxtL = ["." + atmStr for atmStr in unSelectedAtoms]
        selTxt = " ".join(selTxtL)

        nv_view.add_representation(repr_type="label",
                                   selection=selTxt,
                                   labelType="atomname",
                                   color="red")
