#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import MDAnalysis

# @author: melomcr

from . import toolkit as tk
from . import datastorage as ds
import numpy as np
import pandas as pd
from typing import Any


def getCommunityColors() -> pd.DataFrame:
    """Gets standardized colors for communities.

    This function loads pre-specified colors that match those available in
    `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_.

    Returns:
        Returns a pandas dataframe with a VMD-compatible color scale for node
            communities.

    """

    #
    # This function uses the `importlib.resources` package to load a template
    #  file from within the python package.
    # The method that uses `pkg_resources` from `setuptools` introduces
    #  performance overheads.
    #
    # Docs: https://docs.python.org/3.7/library/importlib.html
    from importlib.resources import open_text as pkg_open_text

    from . import vizmod

    # Return a Pandas data frame with a VMD-compatible color scale for node
    # communities.

    with pkg_open_text(vizmod, 'community_RGB_colors.csv') as colorsFileStream:
        commColors = pd.read_csv(colorsFileStream)

    return commColors


def prepTclViz(base_name: str, num_winds: int, ligand_segid: str = "NULL",
               trg_dir: str = "./") -> None:
    """Prepares system-specific TCL script vor visualization.

    This function prepares a TCL script that can be loaded by
    `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_ to create high-resolution
    renderings of the system.

    Args:
        base_name (str): Base name for TCL script and data files necessary for
            visualization.
        num_winds (int) : Number of windows created for analysis.
        ligand_segid (str) : Segment ID of ligand residue. This will create
            a special representations for small molecules.
        trg_dir (str) : Directory where TCL and data files will be saved.

    """

    #
    # This function uses the `importlib.resources` package to load a template
    # file from within the python package.
    # The method that uses `pkg_resources` from `setuptools` introduces
    # performance overheads.
    #
    # Docs: https://docs.python.org/3.7/library/importlib.html
    from importlib.resources import read_text as pkg_read_text
    from importlib.resources import path as pkg_path
    import os

    from . import vizmod

    tclFileName = "network_view_2.tcl"

    # Get path to auxiliary TCL files:
    with pkg_path(vizmod, tclFileName) as pathToVizMod:
        pathToTcls = pathToVizMod.parent

    tclTemplateFile = pkg_read_text(vizmod, tclFileName)

    # Replace base name of output files, number of windows, and ligand segid.
    tclTemplateFile = tclTemplateFile.replace('BASENAMETMP', base_name)
    tclTemplateFile = tclTemplateFile.replace('NUMSTEPTMP', str(num_winds))
    tclTemplateFile = tclTemplateFile.replace('LIGANDTMP', ligand_segid)
    # Replace the path to auxiliary TCLs
    tclTemplateFile = tclTemplateFile.replace('PATHTMP', str(pathToTcls))

    # Write new file to local directory
    with open(os.path.join(trg_dir, tclFileName), 'w') as file:
        file.write(tclTemplateFile)

    msgStr = "The file \'{}\' has been saved in the following folder: {}"
    print(msgStr.format(tclFileName, trg_dir))


def viewPath(nvView: Any,
             path: list,
             dists: Any,
             maxDirectDist: float,
             nodesAtmSel: MDAnalysis.AtomGroup,
             win: int = 0,
             opacity: float = 0.75,
             color: str = "green",
             side: str = "both",
             segments: int = 5,
             disableImpostor: bool = True,
             useCylinder: bool = True) -> None:
    """Creates NGLView representation of a path.

    Renders a series of cylinders to represent a network path.
    The maxDirectDist argument is used as a normalization factor to scale
    representations of edges.
    For more details on NGL parameters, see
    `NGLView's documentation <https://nglviewer.org/nglview/latest/api.html>`_.

    Args:
        nvView (Any) : NGLView object.
        path (list) : Sequence of nodes that define a path.
        maxDirectDist (float): Maximum direct distance between nodes in network.
        nodesAtmSel (Any) : MDAnalysis atom selection object.
        win (int) : Window used for representation.
        opacity (float) : Controls edge opacity.
        color (str) : Controls edge color.
        side (str) : Controls edge rendering quality.
        segments (int) : Controls edge rendering quality.
        disableImpostor (bool) : Controls edge rendering quality.
        useCylinder (bool) : Controls edge rendering quality.

    """

    # TODO: Maybe adapt code to use nx?
    #  nx.reconstruct_path(i, j, pathsPred)

    for firstNodeIndx, secondNodeIndx in zip(path, path[1:]):

        fSel = tk.getNGLSelFromNode(firstNodeIndx, nodesAtmSel)
        sSel = tk.getNGLSelFromNode(secondNodeIndx, nodesAtmSel)

        if dists[win, firstNodeIndx, secondNodeIndx] == 0:
            continue

        radius = 0.05 + (0.5*(dists[win, firstNodeIndx, secondNodeIndx]/maxDirectDist))

        nvView.add_representation("distance",
                                  atom_pair=[[fSel, sSel]],
                                  color=color,
                                  label_visible=False,
                                  side=side,
                                  name="link",
                                  use_cylinder=useCylinder,
                                  radial_sements=segments,
                                  radius=radius,
                                  disable_impostor=disableImpostor,
                                  opacity=opacity,
                                  lazy=True)


def showCommunityGlobal(nvView: Any,
                        nodeCommDF: pd.DataFrame,
                        commID: int,
                        window: int,
                        nodesAtmSel: MDAnalysis.AtomGroup,
                        dnad: ds.DNAdata,
                        colorValDict: Any) -> None:
    """Creates NGLView representation of a specified community.

    Renders a series of cylinders to represent all edges in the network that
    connect nodes in the same community. Edges between nodes in different
    communities are not rendered.

    Args:
        nvView (Any) : NGLView object.
        nodeCommDF (Any) : Pandas data frame relating node IDs with their
            communities.
        commID (float): Community ID for the community to be rendered.
        window (int) : Window used for representation.
        nodesAtmSel (Any) : MDAnalysis atom selection object.
        dnad (Any) : Dynamical Network Analysis data object.
        colorValDict (Any) : Dictionary that standardizes community colors.

    """

    nodes = nodeCommDF.loc[nodeCommDF["Window"+str(window)] == commID, "Node"].values
    nodes.sort()

    if len(nodes) == 0:
        warnStr = "No nodes in this community ({0}) and window ({1})!"
        print(warnStr.format(commID, window))

    showEdges = []
    # Loop from first node to next to last node
    for indI, nodeI in enumerate(nodes[0:-1]):
        # Loop from (indI+1) node to last node
        for indJ, nodeJ in enumerate(nodes[indI+1:]):

            # Skip if there is no edge
            if dnad.corrMatAll[window, nodeI, nodeJ] == 0:
                continue

            # Skip if consecutive nodes (makes visualization cheaper)
            resIdDiff = np.abs(nodesAtmSel.atoms[nodeI].resid -
                               nodesAtmSel.atoms[nodeJ].resid)
            segIdDiff = np.abs(nodesAtmSel.atoms[nodeI].segid ==
                               nodesAtmSel.atoms[nodeJ].segid)
            if (resIdDiff == 1) and (segIdDiff):
                continue

            showEdges.append((nodeI, nodeJ))

    for i, j in showEdges:
        viewPath(nvView, [i, j], dnad.distsAll, dnad.maxDirectDist,
                 nodesAtmSel, win=window, opacity=1, color=colorValDict[commID],
                 side="front", segments=1, disableImpostor=False)


def showCommunityByTarget(nvView: Any,
                          nodeCommDF: pd.DataFrame,
                          trgtNodes: list,
                          window: int,
                          nodesAtmSel: MDAnalysis.AtomGroup,
                          dnad: ds.DNAdata,
                          colorValDict: Any) -> None:
    """Creates NGLView representation of edges between selected nodes and
    their contacts.

    Renders a series of cylinders to represent all edges that connect selected
    nodes with other nodes in contact. Only nodes that have been assigned to a
    community are shown, to minimize the occurrence of unstable contacts. Edges
    between nodes in different communities are still rendered, but shown in
    different representations.

    Args:
        nvView (Any) : NGLView object.
        nodeCommDF (Any) : Pandas data frame relating node IDs with their
            communities.
        trgtNodes (list): List of node IDs.
        window (int) : Window used for representation.
        nodesAtmSel (Any) : MDAnalysis atom selection object.
        dnad (Any) : Dynamical Network Analysis data object.
        colorValDict (Any) : Dictionary that standardizes community colors.

    """

    # Get all nodes in communities larger than 1% of nodes.
    allNodes = nodeCommDF["Node"].values

    for nodeI in trgtNodes:
        # Get all nodes connected to target node "I"
        nodeJList = np.where(dnad.corrMatAll[window, nodeI, :] > 0)[0]

        for nodeJ in nodeJList:

            # Check if both nodes have communities assigned
            if (nodeI not in allNodes) or (nodeJ not in allNodes):
                continue

            # Gets communities IDs from the matched communities
            commI = nodeCommDF.loc[(nodeCommDF["Node"] == nodeI),
                                   "Window"+str(window)].values[0]
            commJ = nodeCommDF.loc[(nodeCommDF["Node"] == nodeJ),
                                   "Window"+str(window)].values[0]

            if commI == commJ:
                useCylinder = True
            else:
                useCylinder = False

            viewPath(nvView, [nodeI, nodeJ], dnad.distsAll, dnad.maxDirectDist,
                     nodesAtmSel, win=window, opacity=1,
                     color=colorValDict[commI], side="front", segments=1,
                     disableImpostor=False, useCylinder=useCylinder)


def showCommunityByID(nvView: Any,
                      cDF: pd.DataFrame,
                      clusID: int,
                      system: str,
                      refWindow: int,
                      shapeCounter: Any,
                      nodesAtmSel: MDAnalysis.AtomGroup,
                      colorValDictRGB: Any,
                      trg_system: str,
                      trg_window: int) -> None:
    """Creates NGLView representation of nodes in a community.

    Renders a series of spheres to represent all nodes in the selected community.

    The `system` argument selects one dataset to ve used for the creation of
    standardized representations, which allows comparisons between variants of
    the same system, with mutations or different ligands, for example.

    The `colorValDictRGB` argument relates colors to RGB codes. This allows the
    creation of several independent representations using the same color scheme.

    For examples of formatted data, see Dynamical Network Analysis tutorial.

    Args:
        nvView (Any) : NGLView object.
        cDF (Any) : Pandas data frame relating node IDs with their communities in
                    every analyzed simulation window. This requires a melt format.
        clusID (float) : ID of the community (or cluster) to be rendered.
        system (str) : System used as reference to standardize representation.
        refWindow (int) : Window used as reference to standardize representation.
        shapeCounter (Any) : Auxiliary list to manipulate NGLView representations.
        nodesAtmSel (Any) : MDAnalysis atom selection object.
        colorValDictRGB (Any) : Dictionary that standardizes community colors.
        trg_system (str) : System to be used for rendering.
        trg_window (int) : Window used for rendering.

    """

    # Displays all nodes of a cluster and colors them by cluster ID.

    nodeList = cDF.loc[(cDF.system == system) &
                       (cDF.Window == refWindow) &
                       (cDF.Cluster == clusID)].Node.values

    for node in nodeList:
        nodeLab = cDF.loc[(cDF.system == trg_system) &
                          (cDF.Window == trg_window) &
                          (cDF.Node == node)].resid.values[0]

        nodePos = list(nodesAtmSel[node].position)
        nodeCol = [x/255 for x in colorValDictRGB[clusID]]
        nvView.shape.add_sphere(nodePos,
                                nodeCol,
                                0.6,
                                nodeLab.split("_")[0] + " ({0})".format(clusID))
        shapeCounter[0] += 1


def showCommunityByNodes(nvView: Any,
                         cDF: pd.DataFrame,
                         nodeList: list,
                         system: str,
                         refWindow: int,
                         shapeCounter: Any,
                         nodesAtmSel: MDAnalysis.AtomGroup,
                         colorValDictRGB: Any) -> None:
    """Creates NGLView representation of nodes in a community.

    Renders a series of spheres to represent all nodes in the selected community.

    Args:
        nvView (Any) : NGLView object.
        cDF (Any) : Pandas data frame relating node IDs with their communities in
                    every analyzed simulation window. This requires a melt format.
        nodeList (list) : List of node IDs to be rendered.
        system (str) : System used as reference to standardize representation.
        refWindow (int) : Window used as reference to standardize representation.
        shapeCounter (Any) : Auxiliary list to manipulate NGLView representations.
        nodesAtmSel (Any) : MDAnalysis atom selection object.
        colorValDictRGB (Any) : Dictionary that standardizes community colors.

    """

    # Displays a given list of nodes and colors them by cluster ID.

    for node in nodeList:
        clusID = cDF.loc[(cDF.system == system) &
                         (cDF.Window == refWindow) &
                         (cDF.Node == node)].Cluster.values[0]

        nodeLab = cDF.loc[(cDF.system == system) &
                          (cDF.Window == refWindow) &
                          (cDF.Node == node)].resid.values[0]

        nodePos = list(nodesAtmSel[node].position)
        nodeCol = [x/255 for x in colorValDictRGB[clusID]]
        nvView.shape.add_sphere(nodePos,
                                nodeCol,
                                0.6,
                                nodeLab.split("_")[0] + " ({0})".format(clusID))
        shapeCounter[0] += 1
