#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr

from . import toolkit as tk

def getCommunityColors():
    '''Gets standardized colors for communities.
    
    This function loads pre-specified colors that match those available in `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_.
    
    Returns:
        Returns a pandas dataframe with a VMD-compatible color scale for node communities.
    
    '''
    
    #
    # This function uses the `importlib.resources` package to load a template file from within the python package.
    # The method that uses `pkg_resources` from `setuptools` introduces performance overheads.
    #
    # Docs: https://docs.python.org/3.7/library/importlib.html?highlight=importlib#module-importlib.resources
    import importlib.resources as pkg_resources
    import pandas as pd
    
    from . import vizmod
    
    colorsFileStream = pkg_resources.open_text(vizmod, 'community_RGB_colors.csv')
    
    # We return a Pandas data frame with a VMD-compatible color scale for node communities.
    return pd.read_csv(colorsFileStream)

def prepTclViz(baseName, numWinds, ligandSegid, trgDir):
    '''Prepares system-specific TCL script vor visualization.
    
    This function prepares a TCL script that can be loaded by `VMD <http://www.ks.uiuc.edu/Research/vmd/>`_ to create high-resolution renderings of the system.
    
    Args:
        baseName (str): Base name for TCL script and data files necessary for visualization.
        numWinds (int) : Number of windows created for analysis.
        ligandSegid (str) : Segment ID of ligand residue. This will create special representations for small molecules.
        trgDir (str) : Directory where TCL and data files will be saved.
    
    '''
    
    #
    # This function uses the `importlib.resources` package to load a template file from within the python package.
    # The method that uses `pkg_resources` from `setuptools` introduces performance overheads.
    #
    # Docs: https://docs.python.org/3.7/library/importlib.html?highlight=importlib#module-importlib.resources
    import importlib.resources as pkg_resources
    from importlib.resources import path as irpath
    import os 
    
    from . import vizmod
    
    # Get path to auxiliary TCL files:
    with irpath(vizmod,"network_view_2.tcl") as pathToVizMod:
        pathToTcls = pathToVizMod.parent
    
    tcvTemplateFile = pkg_resources.read_text(vizmod, 'network_view_2.tcl')
    
    # Read in the file
    #with open(tcvTemplateFile, 'r') as file :
        #tclTemplate = file.read()

    # Replace base name of output files, number of windows, and ligand segid.
    tcvTemplateFile = tcvTemplateFile.replace('BASENAMETMP', baseName)
    tcvTemplateFile = tcvTemplateFile.replace('NUMSTEPTMP', numWinds)
    tcvTemplateFile = tcvTemplateFile.replace('LIGANDTMP', ligandSegid)
    # Replace the path to auxiliary TCLs
    tcvTemplateFile = tcvTemplateFile.replace('PATHTMP', str(pathToTcls) )

    # Write new file to local directory
    with open(os.path.join(trgDir,'network_view_2.tcl'), 'w') as file:
        file.write(tcvTemplateFile)
    
    print("The file \'network_view_2.tcl\' has been saved in the following folder: {}".format(trgDir) )
    

def viewPath(nvView, path, dists, maxDirectDist, nodesAtmSel, win = 0, opacity=0.75, 
             color="green", side="both", segments=5, disableImpostor=True, useCylinder=True):
    '''Creates NGLView representation of a path.
    
    Renders a series of cylinders to represent a network path.
    
    Args:
        nvView (obj) : NGLView object.
        path (list) : Seqeunce of nodes that define a path.
        maxDirectDist (float): Maximum direct distance between nodes in network. Used as a normalization factor to scale representations of edges.
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        win (int) : Window used for representation.
        opacity (float) : Controls edge opacity.
        color (str) : Controls edge color.
        side (str) : Controls edge rendering quality. See `NGLView's documentation <http://nglviewer.org/nglview/latest/api.html>`_.
        segments (int) : Controls edge rendering quality. See `NGLView's documentation <http://nglviewer.org/nglview/latest/api.html>`_.
        disableImpostor (bool) : Controls edge rendering quality. See `NGLView's documentation <http://nglviewer.org/nglview/latest/api.html>`_.
        useCylinder (bool) : Controls edge rendering quality. See `NGLView's documentation <http://nglviewer.org/nglview/latest/api.html>`_.
    
    '''
    
    # Maybe adapt code to use nx?
    #nx.reconstruct_path(i, j, pathsPred)
    
    for firstNodeIndx, secondNodeIndx in zip(path, path[1:]):
        
        fSel = tk.getNGLSelFromNode(firstNodeIndx, nodesAtmSel)
        sSel = tk.getNGLSelFromNode(secondNodeIndx, nodesAtmSel)
        
        if dists[win, firstNodeIndx, secondNodeIndx] == 0:
            continue
        
        radius = 0.05 + (0.5*(dists[win, firstNodeIndx, secondNodeIndx]/maxDirectDist))
        
        nvView.add_representation("distance",atom_pair=[[fSel,sSel]], color=color, label_visible=False, 
                            side=side, name="link", use_cylinder=useCylinder, radial_sements=segments, 
                            radius=radius, disable_impostor=disableImpostor, opacity=opacity, lazy=True)
    

def showCommunity(nvView, nodeCommDF, commID, window, nodesAtmSel, dnad, colorValDict):
    '''Creates NGLView representation of a specified community.
    
    Renders a series of cylinders to represent all edges in the network that connect nodes in the same community. Edges between nodes in different communities are not rendered.
    
    Args:
        nvView (obj) : NGLView object.
        nodeCommDF (obj) : Pandas data frame relating node IDs with their communities in every analyzed simulation window. See Dynamical Network Analysis tutorial.
        commID (float): Community ID for the community to be rendered.
        window (int) : Window used for representation.
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        dnad (obj) : Dynamical Network Analysis data object.
        colorValDict (obj) : Dictionary that standardizes community colors.
    
    '''
    
    # Gets the list of all connected pairs in the system
    nonzeroPairs = [(i,j) for i,j in np.asarray(np.where(dnad.corrMatAll[window ,:,:] > 0)).T if i < j]
    
    for i,j in nonzeroPairs:
        commI = dnad.nxGraphs[window].nodes[i]["modularity"]
        commJ = dnad.nxGraphs[window].nodes[j]["modularity"]
        
        # If the pair of nodes is in the same cmmunity, render the edge.
        if commID == commI == commJ :
            viewPath(nvView, [i,j], dnad.distsAll, dnad.maxDirectDist, 
                     nodesAtmSel, win = window, opacity=1, color=colorValDict[commID])

def showCommunityGlobal(nvView, nodeCommDF, commID, window, nodesAtmSel, dnad, colorValDict):
    '''Creates NGLView representation of a specified community.
    
    This is a variation of :py:func:`showCommunity` created for reducing the quality and therefore the cost of creating edge representations. It is particularly useful in very large systems. 
    
    Renders a series of cylinders to represent all edges in the network that connect nodes in the same community. Edges between nodes in different communities are not rendered.
    
    Args:
        nvView (obj) : NGLView object.
        nodeCommDF (obj) : Pandas data frame relating node IDs with their communities in every analyzed simulation window. See Dynamical Network Analysis tutorial.
        commID (float): Community ID for the community to be rendered.
        window (int) : Window used for representation.
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        dnad (obj) : Dynamical Network Analysis data object.
        colorValDict (obj) : Dictionary that standardizes community colors.
    
    '''
    
    nodes = nodeCommDF.loc[ nodeCommDF["Window"+str(window)] == commID, "Node" ].values
    nodes.sort() #Sort in place
    
    if len(nodes) == 0:
        print("No nodes in this community ({0}) and window ({1})!".format(commID, window))
    
    showEdges = []
    # Loop from fisrt node to next to last node
    for indI,nodeI in enumerate(nodes[0:-1]):
        # Loop from (indI+1) node to last node
        for indJ,nodeJ in enumerate(nodes[indI+1:]):
            # Skip if there is no edge
            if dnad.corrMatAll[window , nodeI, nodeJ] == 0:
                continue
            # Skip if consecutive nodes (makes visualization cheaper)
            if ((np.abs(nodesAtmSel.atoms[nodeI].resid - nodesAtmSel.atoms[nodeJ].resid) == 1) and
                  (np.abs(nodesAtmSel.atoms[nodeI].segid == nodesAtmSel.atoms[nodeJ].segid) ) ):
                continue
            showEdges.append((nodeI, nodeJ))
    
    for i,j in showEdges:
#         viewPath(nvView, [i,j], dnad.distsAll, dnad.maxDirectDist, 
#                  nodesAtmSel, win = window, opacity=1, color=colorValues[commID])
        viewPath(nvView, [i,j], dnad.distsAll, dnad.maxDirectDist, 
             nodesAtmSel, win = window, opacity=1, color=colorValDict[commID], 
             side="front", segments=1, disableImpostor=False)

def showCommunityByTarget(nvView, nodeCommDF, trgtNodes, window, nodesAtmSel, dnad, colorValDict):
    '''Creates NGLView representation of edges between selected nodes and their contacts.
    
    Renders a series of cylinders to represent all edges that connect selected nodes with other nodes in contact. Only nodes that have been assigned to a community are shown, to minimize the occurence of unstable contacts.
    Edges between nodes in different communities are still rendered, but shown in different representations.
    
    Args:
        nvView (obj) : NGLView object.
        nodeCommDF (obj) : Pandas data frame relating node IDs with their communities in every analyzed simulation window. See Dynamical Network Analysis tutorial.
        trgtNodes (list): List of node IDs.
        window (int) : Window used for representation.
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        dnad (obj) : Dynamical Network Analysis data object.
        colorValDict (obj) : Dictionary that standardizes community colors.
    
    '''
    
    import numpy as np
    
    #Get all nodes in cummunities larger than 1% of nodes.
    allNodes = nodeCommDF["Node" ].values
    
    for nodeI in trgtNodes:
        # Get all nodes connected to target node "I"
        nodeJList = np.where( dnad.corrMatAll[window, nodeI, :] > 0 )[0]
        
        for nodeJ in nodeJList :
            
            # Check if both nodes have communities assigned
            if (not nodeI in allNodes) or (not nodeJ in allNodes):
                continue
            
            # Gets communities IDs from the matched communities
            commI = nodeCommDF.loc[ ( nodeCommDF["Node"] == nodeI ), "Window"+str(window) ].values[0]
            commJ = nodeCommDF.loc[ ( nodeCommDF["Node"] == nodeJ ), "Window"+str(window) ].values[0]
            
            if commI == commJ:
                useCylinder = True
            else:
                useCylinder = False
            
            viewPath(nvView, [nodeI, nodeJ], dnad.distsAll, dnad.maxDirectDist, 
                     nodesAtmSel, win = window, opacity=1, color=colorValDict[commI], 
                     side="front", segments=1, disableImpostor=False, useCylinder=useCylinder)

def showCommunityByID(nvView, cDF, clusID, system, refWindow, shapeCounter, nodesAtmSel, colorValDictRGB, system1, refWindow1):
    '''Creates NGLView representation of nodes in a community.
    
    Renders a series of spheres to represent all nodes in the selected community.
    
    Args:
        nvView (obj) : NGLView object.
        cDF (obj) : Pandas data frame relating node IDs with their communities in every analyzed simulation window. This requires a melt format. See Dynamical Network Analysis tutorial.
        clusID (float) : ID of the community (or cluster) to be rendered.
        system (str) : System used as reference to standardize representation. This allows comparisons between variants of the same system, with mutations or different ligands.
        refWindow (int) : Window used as reference to standardize representation.
        shapeCounter (obj) : Auxiliary list used to help manipulate NGLView representations. 
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        colorValDictRGB (obj) : Dictionary that standardizes community colors. Relates colors to RGB codes.
        system1 (str) : System to be used for rendering. This allows comparisons between variants of the same system, with mutations or different ligands.
        refWindow1 (int) : Window used for rendering.
    
    '''
    
    # Dysplays all nodes of a cluster and colors them by cluster ID.
    
    nodeList = cDF.loc[ (cDF.system == system) & (cDF.Window == refWindow) & (cDF.Cluster == clusID) ].Node.values
    for node in nodeList:
        nodeLab = cDF.loc[ (cDF.system == system1) & (cDF.Window == refWindow1) & (cDF.Node == node) ].resid.values[0]
        nodePos = list(nodesAtmSel[node].position)
        nodeCol = [x/255 for x in colorValDictRGB[clusID]]
        nvView.shape.add_sphere( nodePos, nodeCol, 0.6, nodeLab.split("_")[0] + " ({0})".format(clusID) )
        shapeCounter[0] += 1

def showCommunityByNodes(nvView, cDF, nodeList, system, refWindow, shapeCounter, nodesAtmSel, colorValDictRGB):
    '''Creates NGLView representation of nodes in a community.
    
    Renders a series of spheres to represent all nodes in the selected community.
    
    Args:
        nvView (obj) : NGLView object.
        cDF (obj) : Pandas data frame relating node IDs with their communities in every analyzed simulation window. This requires a melt format. See Dynamical Network Analysis tutorial.
        nodeList (list) : List of node IDs to be rendered.
        system (str) : System used as reference to standardize representation. This allows comparisons between variants of the same system, with mutations or different ligands.
        refWindow (int) : Window used as reference to standardize representation.
        shapeCounter (obj) : Auxiliary list used to help manipulate NGLView representations. 
        nodesAtmSel (obj) : MDAnalysis atom selection object.
        colorValDictRGB (obj) : Dictionary that standardizes community colors. Relates colors to RGB codes.
        
    '''
    
    # Dysplays a given list of nodes and colors them by cluster ID.
    
    for node in nodeList:
        clusID = cDF.loc[ (cDF.system == system) & (cDF.Window == refWindow) & (cDF.Node == node) ].Cluster.values[0]
        nodeLab = cDF.loc[ (cDF.system == system) & (cDF.Window == refWindow) & (cDF.Node == node) ].resid.values[0]
        nodePos = list(nodesAtmSel[node].position)
        nodeCol = [x/255 for x in colorValDictRGB[clusID]]
        nvView.shape.add_sphere( nodePos, nodeCol, 0.6, nodeLab.split("_")[0] + " ({0})".format(clusID) )
        shapeCounter[0] += 1
