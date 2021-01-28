#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr


from . import toolkit as tk
from . import gencor as gc
from . import contact as ct
from . import network as nw
from . import datastorage as ds

import numpy as np

import cython

import MDAnalysis as mda

import multiprocessing as mp

import networkx as nx

import community

from networkx                       import eigenvector_centrality_numpy as nxeigencentrality
from networkx                       import edge_betweenness_centrality as nxbetweenness
from networkx.algorithms.shortest_paths.dense import floyd_warshall_predecessor_and_distance as nxFWPD

from MDAnalysis.lib.NeighborSearch  import AtomNeighborSearch   as mdaANS
from MDAnalysis.coordinates.memory  import MemoryReader         as mdaMemRead
from MDAnalysis.analysis.base       import AnalysisFromFunction as mdaAFF
from MDAnalysis.analysis            import distances            as mdadist 
from MDAnalysis.analysis.distances  import between              as mdaB

from colorama import Fore
from operator       import itemgetter
from collections    import OrderedDict, defaultdict

import copy,os

##################################################
##################################################

class DNAproc:
    '''The Dynamic Network Analysis processing class contains the 
    infrastructure to carry out the data analysis in a DNA study 
    of a biomolecular system.
    
    This class uses optimized auxiliary functions to parallelize the most 
    time-consuming aspects of Dynamical Network Analysis, including 
    contact detection, calculation of correlation coefficients, and
    network properties.
    
    The infrastructure built here is also focused on combining multiple
    molecular dynamics simulations of the same system, emphasizing the 
    calculation of statistical properties. This allows the comparison 
    between replicas of the same biomolecular system or time evolution 
    of a particular system.
    
    '''
    
    def __init__(self):
        '''Constructor
        
        Sets initial values for class variables.
        
        '''
        
        self.dnaData = None 
        
        self.contactPersistence = 0
        self.cutoffDist = 0
        self.numSampledFrames = 0
        self.numWinds = 0
        
        # Number of neighbours for Generalized Correlation estimate
        self.kNeighb = 7
        
        self.h2oName            = None
        self.segIDs             = None
        self.customResNodes     = None
        self.usrNodeGroups      = None
        
        self.allResNamesSet     = None
        self.selResNamesSet     = None
        self.notSelResNamesSet  = None
        self.notSelSegidSet     = None
        
        self.workU          = None
        
        self.selRes         = None
        
        self.nodesAtmSel    = None
        self.atomToNode     = None
        self.resNodeGroups  = None
        
        self.numNodes       = None
        
        self.nodeGroupIndicesNP     = None
        self.nodeGroupIndicesNPAux  = None
        self.contactMatAll          = None
        self.corrMatAll             = None
        
        self.nodeDists              = None
        self.nxGraphs               = None
        self.distsAll               = None
        self.preds                  = None
        self.maxDist                = None
        self.maxDirectDist          = None
        self.btws                   = None
        self.nodesComm              = None
        
        self.interNodePairs         = None
        self.contactNodesInter      = None
        
    def setNumWinds(self, numWinds):
        '''Set number of windows.
        
        Args:
            numWinds (int) : Number of windows into which the trajectory will be split.
        '''
        
        self.numWinds = numWinds
    
    def setNumSampledFrames(self, numSampledFrames):
        '''Set number of frames to be sampled for solvent detection.
        
        Args:
            numSampledFrames (int) : Number of frames sampled for solvent detection and for estimation of cartesian distance between node groups.
        '''
        
        self.numSampledFrames = numSampledFrames
    
    def setCutoffDist(self, cutoffDist):
        '''Set cartesian distance cutoff for contact detection.
        
        Args:
            cutoffDist (float) : Cutoff distance for contact detection. Usually set to 4.5 Angstroms.
        
        '''
        self.cutoffDist = cutoffDist
        
    def setContactPersistence(self, contactPersistence):
        '''Set contact persistance cutoff for contact detection.
        
        Args:
            contactPersistence (float) : Ratio of total trajectory frames needed to consider a pair of nodes to be in contact. Usually set to 0.75 (75% of total trajectory).
        
        '''
        
        self.contactPersistence = contactPersistence
        
    def seth2oName(self, h2oName):
        '''Set name of solvent molecule residue.
        
        Args:
            h2oName (list) : List of residue names used as solvent.
        
        '''
        
        self.h2oName = h2oName
        
    def setSegIDs(self, segIDs):
        '''Set segment IDs for biomolecules ot be analyzed.
        
        Args:
            segIDs (list) : List of Segment IDs to be included in network analysis.
        
        '''
        
        self.segIDs = segIDs
    
    def setCustomResNodes(self, customResNodes):
        '''Set atoms that will represent nodes in user defined residues.
        
        Network Analysis will create one network node per standard amino acid residue (in the alpha carbon). For other residues, the user must specify atom(s) that will represent a node.
        This function is used to define which residue atoms will be used to create network nodes.
        
        .. note:: See also :py:func:`setUsrNodeGroups`.
        
        Args:
            customResNodes (dictionary) : Dictionary mapping residue names with lists of atom names that will represent network nodes.
        
        '''
        
        self.customResNodes = customResNodes
    
    def setUsrNodeGroups(self, usrNodeGroups):
        '''Set atoms that will represent node groups in user-defined residues.
        
        Network Analysis will create one network node per standard amino acid residue (in the alpha carbon). For other residues, the user must specify atom(s) that will represent a node.
        This function is used to define the heavy atoms that compose each node group for user-defined nodes.
        
        .. note:: See also :py:func:`setCustomResNodes`.
        
        Args:
            usrNodeGroups (dictionary) : Nested dictionary mapping residue names with atom names that will represent network nodes, and sets of heavy atoms used to define node groups.
        
        '''
        
        self.usrNodeGroups = usrNodeGroups
    
    def getU(self):
        '''Return MDAnalysis universe object.
        '''
        return self.workU 
    
    def saveData(self, fileNameRoot="dnaData"):
        '''Save all network analysis data to file.
        
        This function automates the creation of a :py:class:`~dynetan.datastorage.DNAdata` object, the placement of data in the object, and the call to its :py:func:`~dynetan.datastorage.DNAdata.saveToFile` function.
        
        Args:
            fileNameRoot (str) : Root of the multiple data files to be writen.
        
        '''
        
        self.dnaData = ds.DNAdata()
        
        self.dnaData.nodesIxArray   = self.nodesAtmSel.ix_array
        self.dnaData.numNodes       = self.numNodes
        self.dnaData.atomToNode     = self.atomToNode
        self.dnaData.contactMat     = self.contactMat
        
        self.dnaData.corrMatAll     = self.corrMatAll
        self.dnaData.nodeDists      = self.nodeDists
        
        self.dnaData.distsAll       = self.distsAll
        self.dnaData.preds          = self.preds
        self.dnaData.maxDist        = self.maxDist
        self.dnaData.maxDirectDist  = self.maxDirectDist
        self.dnaData.btws           = self.btws
        self.dnaData.nodesComm      = self.nodesComm
        self.dnaData.nxGraphs       = self.nxGraphs
        
        self.dnaData.interNodePairs    = self.interNodePairs
        self.dnaData.contactNodesInter = self.contactNodesInter
        
        self.dnaData.saveToFile(fileNameRoot)
        
    def saveReducedTraj(self, fileNameRoot="dnaData", stride=1):
        '''Save a reduced trajectory to file.
        
        This function automates the creation of a reduced DCD trajectory file keeping only the atoms used for Dynamical Network Analysis. It also creates a matching PDB file to maintain atom and residue names.
        
        Args:
            fileNameRoot (str) : Root of the trajectory and structure files to be writen.
            stride (int) : Stride used to write the trajectory file.
        
        '''
        
        dcdVizFile = fileNameRoot + "_reducedTraj.dcd"
        
        with mda.Writer(dcdVizFile, self.workU.atoms.n_atoms) as W:
            for ts in tk.log_progress(self.workU.trajectory[::stride], 
                                      every=1, 
                                      size=int(len(self.workU.trajectory[::stride])),
                                name="Frames"):
                W.write(self.workU.atoms)
        
        pdbVizFile = fileNameRoot + "_reducedTraj.pdb"

        with mda.Writer(pdbVizFile, multiframe=False, bonds="conect", n_atoms=self.workU.atoms.n_atoms) as PDB:
            PDB.write(self.workU.atoms)
        
    def loadSystem(self, psfFile, dcdFiles):
        '''Loads PSF and DCD files to an MDAnalysis universe.
        
        '''
        self.workU = mda.Universe(psfFile,dcdFiles)

    def checkSystem(self):
        '''Performs a series of sanity checks.
        
        This function checks if the user-defined data and loaded symulation data are complete and compatible. This will print a series of diagnostic messages that should be used to verify if all calculations are set up as desired.
        
        '''
        
        if not self.workU:
            print("ERROR! This function can only be called after loading a system. Check your universe!")
            return -1
        
        allResNamesSet = set()
        selResNamesSet = set()
        notSelResNamesSet = set()
        notSelSegidSet = set()
        
        print(Fore.BLUE + "Residue verification:\n" + Fore.RESET)

        # Loop over segments and checks residue names
        for segment in self.workU.segments:
            segid = segment.segid
            
            resNames = set([res.resname for res in segment.residues ])
            
            if segid in self.segIDs:
                print("---> SegID ",Fore.GREEN + segid, Fore.RESET +":", len(resNames),"unique residue types:")
                print(resNames)
                print()
                selResNamesSet.update(resNames)
            else:
                notSelSegidSet.add(segid)
                
            allResNamesSet.update(resNames)
            
        print("---> {0} total selected residue types:".format(len(selResNamesSet)))
        print(selResNamesSet) ; print()

        notSelResNamesSet = allResNamesSet - selResNamesSet

        print(("---> {0} " + Fore.RED + "not-selected" + Fore.RESET + \
            " residue types in other segments:").format(len(notSelResNamesSet)))
        print(notSelResNamesSet)  ; print()

        print("---> {0} total residue types:".format(len(allResNamesSet)))
        print(allResNamesSet)  ; print()

        self.selRes = self.workU.select_atoms("segid " + " ".join(self.segIDs))
        print("---> " + Fore.GREEN + "{0} total residues".format(len(self.selRes.residues))
              + Fore.RESET + " were selected for network analysis.") ; print()

        print(Fore.BLUE + "Segments verification:\n" + Fore.RESET)

        print(("---> {0} " + Fore.GREEN + "selected" + Fore.RESET + " segments:").format(len(self.segIDs)))
        print(self.segIDs)  ; print()

        print(("---> {0} " + Fore.RED + "not-selected" + Fore.RESET + " segments:").format(len(notSelSegidSet)))
        print(sorted(notSelSegidSet, key=str.lower))  ; print()
        
        self.allResNamesSet = allResNamesSet
        self.selResNamesSet = selResNamesSet
        self.notSelResNamesSet = notSelResNamesSet
        self.notSelSegidSet = notSelSegidSet
    
    def selectSystem(self, withSolvent=False, userSelStr=None, verbose=0):
        '''Selects all atoms used to define node groups.
        
        Creates a final selection of atoms based on the user-defined residues and
        node groups. This function also automates solvent and ion detection, for 
        residues that make significant contacts with network nodes. Examples are 
        structural water molecules and ions.
        
        This function will automatically remove all hydrogen atoms from the system, 
        since they are not used to detect contacts or to calculate correlations.
        The standard selection string used is "not (name H* or name [123]H*)"
        
        Ultimately, an MDAnalysis universe is created with the necessary simulation data, reducing the amount of memory used by subsequent analysis.
        
        Args:
            
            withSolvent (bool): Controls if the function will try to automatically detect solvent molecules.
            
            userSelStr (str): Uses a user-defined seletion for the system. This disables automatic detection of solvent/ions/lipids and other residues that may have transient contact with the target system.
        '''
        
        if userSelStr:
            print("Using user-defined selection string:")
            print(userSelStr)
            print("\nATTENTION: automatic identification of solvent and ions is DISABLED.")
            
            initialSel = self.workU.select_atoms(userSelStr)
            
        else:
            if withSolvent:
                if self.notSelSegidSet:
                    checkSet = self.workU.select_atoms("(not (name H* or name [123]H*)) and segid " + " ".join(self.notSelSegidSet) )
                else:
                    print("WARNING: All segments have been selected for Network Analysis, none are left for automatic identification of structural solvent molecules or lipids.")
                    checkSet = None
            else:
                if self.notSelSegidSet:
                    checkSet = self.workU.select_atoms("segid " + " ".join(self.notSelSegidSet) + \
                        " and not resname " + " ".join(self.h2oName) + " and not (name H* or name [123]H*)")
                else:
                    print("WARNING: All segments have been selected for Network Analysis, none are left for automatic identification of transient contacts.")
                    checkSet = None
            
            if checkSet:
                numAutoFrames = self.numSampledFrames*self.numWinds

                stride = int(np.floor(len(self.workU.trajectory)/numAutoFrames))

                print("Checking {0} frames (striding {1})...".format(numAutoFrames, stride))

                searchSelRes = self.selRes.select_atoms("not (name H* or name [123]H*)")

                # Keeps a set with all residues that were close to the interaction region in ALL
                #  sampled timesteps

                resIndexDict = defaultdict(int)
                for ts in tk.log_progress(self.workU.trajectory[:numAutoFrames*stride:stride], name="Frames",size=numAutoFrames):
                    
                    # Creates neighbor search object. We pass the atoms we want to check,
                    #   and then search using the main selection.
                    # This is expensive because it creates a KD-tree for every frame, 
                    #   but the search for Neighbors is VERY fast.
                    searchNeigh = mdaANS(checkSet)
                    
                    resNeigh = searchNeigh.search(searchSelRes, self.cutoffDist)
                    
                    for indx in resNeigh.residues.ix:
                        resIndexDict[indx] += 1
                    

                resIndxList = [ k for k,v in resIndexDict.items() if v > int(numAutoFrames*self.contactPersistence) ]

                checkSetMin = self.workU.residues[ np.asarray(resIndxList, dtype=int) ]

                print("{} extra residues will be added to the system.".format(len(checkSetMin.resnames)))
                
                if verbose:
                    print("New residue types included in the system selection:")
                    for resname in set(checkSetMin.resnames):
                        print(resname)
                    print("New residues included in the system selection:")
                    for res in set(checkSetMin.residues):
                        print(res)
                
                selStr = "segid " + " ".join(self.segIDs) 
                initialSel = self.workU.select_atoms(selStr)
                initialSel = initialSel.union(checkSetMin.atoms)
                initialSel = initialSel.select_atoms("not (name H* or name [123]H*)")
                
            else:
                # In case we do not have any residues in other segments to check for contacts,
                # we take all user-selected segments and create the system for analysis.
                selStr = "segid " + " ".join(self.segIDs) 
                initialSel = self.workU.select_atoms(selStr)
                initialSel = self.workU.select_atoms("not (name H* or name [123]H*)")
        
        print("The initial universe had {} atoms.".format( len(self.workU.atoms) ))
        
        # Merging a selection from the universe returns a new (and smaller) universe
        self.workU = mda.core.universe.Merge(initialSel)
        
        print("The final universe has {} atoms.".format( len(self.workU.atoms) ))
        
        # We now load the new universe to memory, with coordinates from the selected residues.
        
        print("Loading universe to memory...")

        self.workU.load_new(mdaAFF(lambda ag: ag.positions.copy(),
                                    initialSel).run().results, 
                    format=mdaMemRead)
        self.workU
        
    def prepareNetwork(self, verbose=0):
        '''Prepare network representation of the system.
        
        Checks if we know how to treat all types of residues in the final system selection. Every residue will generate one or more nodes in the final network. This function also processes and stores the groups of atoms that define each node group in specialized data structures.
        
        .. note:: We need this special treatment because the residue information in the topology file may list atoms in an order that separates atoms from the same node group. Even though atoms belonging to the same residue are contiguous, atoms in our arbitrary node groups need not be contiguous. Since amino acids have just one node, they will have just one range of atoms but nucleotides and other residues may be different.
        
        '''
        
        from itertools import groupby
        from itertools import chain
        from operator import itemgetter

        # Dictionary associating residue type and node-atom to set of 
        #   atom names associated with that node.
        self.resNodeGroups = {}

        self.resNodeGroups.update(self.usrNodeGroups)

        for res in self.workU.select_atoms("not protein").residues:
            # Verifies if there are unkown residues
            if len(res.atoms) > 1 and res.resname not in self.customResNodes.keys():
                print((Fore.RED + "Unknown residue type" + Fore.RESET + " {0}, from segment {1}").format(
                    res.resname, res.segid))
            
            # For residues that are not proteic, and that have one atom (IONS)
            # Creates the "node group" and the atom name for the node.
            if len(res.atoms) == 1:
                self.resNodeGroups[res.resname] = {}
                self.resNodeGroups[res.resname][res.atoms.names[0]] = set(res.atoms.names)

                if res.resname not in self.customResNodes.keys():
                    self.customResNodes[res.resname] = [res.atoms[0].name]
            else:
                # If the residue is not an ION, check for Hydrogen atoms.
                
                # Adds hydrogen atoms to a groups of atoms in every residue.
                for atm in res.atoms:
                    # Assume it is a hydrogen and bind it to the group of the atom
                    # it is connected to.
                    if atm.name not in set.union(*self.resNodeGroups[res.resname].values()):
                        boundTo = atm.bonded_atoms[0].name
                        
                        for key, val in self.resNodeGroups[res.resname].items():
                            if boundTo in val:
                                self.resNodeGroups[res.resname][key].add(atm.name)
            
            
            kMissing = set(self.resNodeGroups[res.resname].keys()).difference(set(res.atoms.names))
            if kMissing:
                warningStr = (Fore.RED + "Warning!" + Fore.RESET + " residue {0} \
        segid {1} resid {2} does not contain all node atoms. Missing atoms: {3}").format(
                    res.resname, res.segid, res.resid, ' '.join(kMissing))
                print(warningStr)
                
        resNodeAtoms = self.customResNodes.copy()

        # Creates node groups for protein atoms.
        for res in self.workU.select_atoms("protein").residues:
            if res.resname not in resNodeAtoms.keys():
                resNodeAtoms[res.resname] = ["CA"]
                # Creates the group of atoms in the group represented by this node.
                self.resNodeGroups[res.resname] = {}
                self.resNodeGroups[res.resname]["CA"] = set(res.atoms.names)
            else:
                self.resNodeGroups[res.resname]["CA"].update(set(res.atoms.names))

        ## Create atom selection for atoms that represent nodes

        # Builds list of selection statements
        selStr = ["(protein and name CA)"]
        selStr += [ "(resname {0} and name {1})".format(k," ".join(v)) for k,v in self.customResNodes.items() ]
        # Combines all statements into one selection string
        selStr = " or ".join(selStr)
        
        if verbose:
            print("Selection string for atoms that represent network nodes:")
            print(selStr)
        
        self.nodesAtmSel = self.workU.select_atoms(selStr)

        self.numNodes = self.nodesAtmSel.n_atoms
        
        print("Preparing nodes...")
        
        #self.atomToNode = np.full(shape=initialSel.n_atoms, fill_value=-1, dtype=int)
        self.atomToNode = np.full(shape=len(self.workU.atoms), fill_value=-1, dtype=int)
        
        # Creates an array relating atoms to nodes.
        for indx, node in enumerate(tk.log_progress(self.nodesAtmSel.atoms, name="Nodes")):
            
            # Loops over all atoms in the residue related to the node
            for atm in node.residue.atoms:
                
                # Checks if the atom name is listed for the node
                if atm.name in self.resNodeGroups[node.resname][node.name]:
                    self.atomToNode[atm.ix] = indx
                
        # Verification: checks if there are any "-1" left. If so, that atom was not assigned a node.
        loneAtms = np.where( self.atomToNode < 0 )[0]
        if len(loneAtms) > 0:
            print("ERROR: Atoms were not assigned to any node! This can be a problem with your definition of nodes and atom groups.")
            print( "Lone atoms: ")
            print( loneAtms )
            print( "Lone atoms (types and residues): ")
            for atm in loneAtms:
                print( self.workU.atoms[atm] )
        
        # Determine groups of atoms that define each node.
        # We need all this because the topology in the PSF may 
        #   list atoms in an order that separates atoms from the same node.
        #   Even though atoms in the same *residue* are contiguous,
        #   atoms in our arbitrary node groups need not be contiguous.
        # Since amino acids have just one node, they will have just one range
        #   but nucleotides and other residues may be different.

        nodeGroupRanges = {}

        nodeGroupIndices = []

        for x in np.unique(self.atomToNode):
            data = np.where(self.atomToNode == x)[0]
            
            ranges =[]
            for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
                # Creates an iterable from the group object.
                group = (map(itemgetter(1),g))
                # Creates a list using the itterable
                group = list(map(int,group))
                # Appends to the list of ranges the first and last items in each range.
                ranges.append((group[0],group[-1]))
            
            nodeGroupRanges[x] = ranges
            
            # Transforms the ranges into lists
            tmp = [ [x for x in range(rang[0],rang[1]+1)] for rang in nodeGroupRanges[x]]
            # Combine lists into one list
            nodeGroupIndices.append( list(chain.from_iterable(tmp)) )
            
        nodeGroupIndices = np.asarray(nodeGroupIndices, dtype=object)

        self.nodeGroupIndicesNP = np.asarray(list(chain.from_iterable(nodeGroupIndices)), dtype=int)
        
        self.nodeGroupIndicesNPAux = np.zeros(self.numNodes, dtype=int)

        for indx, atmGrp in enumerate(nodeGroupIndices[1:], start=1):
            self.nodeGroupIndicesNPAux[indx] = len(nodeGroupIndices[indx -1]) + self.nodeGroupIndicesNPAux[indx -1]
        
        print("Nodes are ready for network analysis.")
        
        return None
        
    def alignTraj(self, inMemory=True):
        '''Wrapper function for MDAnalysis trajectory alignment tool.
        '''
        from MDAnalysis.analysis import align as mdaAlign
        
        # Set the first frame as reference for alignment
        self.workU.trajectory[0]

        alignment = mdaAlign.AlignTraj(self.workU, self.workU, 
                                       select="segid " + " ".join(self.segIDs) + " and not (name H* or name [123]H*)",
                                       verbose=True, 
                                       in_memory=True, 
                                       weights="mass" )
        alignment.run()
        

    def _contactTraj(self, contactMat, beg = 0, end = -1, stride = 1):
        '''Wrapper for contact calculation per trajectory window.
        
        Pre allocates the necessary temporary NumPy arrays to speed up calculations.
        
        '''
        
        # Creates atom selection for distance calculation
        selectionAtms = self.workU.select_atoms("all")
        
        nAtoms = selectionAtms.n_atoms
        
        # Array to receive all-to-all distances, at every step
        tmpDists = np.zeros( int(nAtoms*(nAtoms-1)/2), dtype=float )
        
        # Array to get minimum distances per node
        tmpDistsAtms = np.full( nAtoms, self.cutoffDist*2, dtype=float )
        
        if end < 0:
            end = self.workU.trajectory.n_frames
        
        for ts in self.workU.trajectory[beg:end:stride]:
            
            # Calculates distances to determine contact matrix
            ct.getContactsC(selectionAtms, self.numNodes, nAtoms, self.cutoffDist, tmpDists, tmpDistsAtms, 
                        contactMat, self.atomToNode, 
                        self.nodeGroupIndicesNP, self.nodeGroupIndicesNPAux)

    def findContacts(self, stride=1):
        '''Finds all nodes in contact.
        
        This is the main user interface access to calculate nodes in contact. This function automatically splits the whole trajectory into windows, allocates NumPy array objects to speed up claculations, and leverages MDAnalysis parallel implementation to determine atom distances.
        
        After determining which frames of a trajectory window show contacts between atom groups, it checks the contact cutoff to determine if two nodes has enough contact during a simulation window to be considered "in contact". A final contact matrix is created and stored in the DNAproc object.
        
        This function automatically updates the unified contact matrix that displays nodes in contact in *any* simulation window.
        The function also performs general sanity checks by calling :py:func:`checkContactMat`.
        
        Args:
            stride (int) : Controls how many trajectory frames will be skipped during contact claculation.
        
        '''
        
        numFrames = self.workU.trajectory.n_frames
        
        # Allocate contact matrix(ces)
        self.contactMatAll = np.zeros([self.numWinds, self.numNodes, self.numNodes], dtype=np.int)
        
        # Set number of frames per window.
        winLen = int(np.floor(self.workU.trajectory.n_frames/self.numWinds))

        # Set number of frames that defines a contact
        contactCutoff = (winLen/stride)*self.contactPersistence

        for winIndx in tk.log_progress(range(self.numWinds),every=1, size=self.numWinds, name="Window"):
            beg = winIndx*winLen
            end = (winIndx+1)*winLen
            
            ## RUNs calculation
            self._contactTraj( self.contactMatAll[winIndx, :, :], beg, end, stride)

            self.contactMatAll[winIndx, :, :] = np.where(self.contactMatAll[winIndx, :, :] > contactCutoff, 1, 0)

            for i in range(self.numNodes):
                for j in range(i+1, self.numNodes):
                    self.contactMatAll[winIndx, j,i] = self.contactMatAll[winIndx, i,j]
        
        # Update unified contact matrix for all windows
        self._genContactMatrix()
        
        self.checkContactMat()
    
    def _genContactMatrix(self):
        '''Update unified contact matrix for all windows.
        '''
        # Allocate a unified contact matrix for all windows.
        self.contactMat = np.zeros([self.numNodes, self.numNodes], dtype=np.int)
        
        # Join all data for all windows on the same unified contact matrix.
        for winIndx in range(self.numWinds):
            self.contactMat = np.add(self.contactMat, self.contactMatAll[winIndx, :, :])
        
        # Creates binary mask with pairs of nodes in contact
        self.contactMat = np.where(self.contactMat > 0, 1, 0)
    
    def checkContactMat(self, verbose=True):
        '''Sanity checks for contact matrix for all windows.
        
        Checks if the contact matrix is symmetric and if there are any nodes that make no contacts to any other nodes across all windows.
        The function also calculates the percentage of nodes in contact over the entire system.
        
        Args:
            verbose (bool) : Controls how much output will the function print.
        
        '''
        
        # Sanity Checks that the contact matrix is symmetric
        if not np.allclose(self.contactMat, self.contactMat.T, atol=0.1):
            print("ERROR: the contact matrix is not symmetric.")
        
        # Checks if there is any node that does not make contacts to ANY other node.
        noContactNodes = np.asarray(np.where( np.sum(self.contactMat, axis=1) == 0 )[0])
        if verbose: 
            print("We found {0} nodes with no contacts.".format(len(noContactNodes)))
        
        # Counts total number of contacts
        if verbose: 
            pairs = np.asarray(np.where(np.triu(self.contactMat) > 0)).T
            totalPairs = int(self.contactMat.shape[0]*(self.contactMat.shape[0]-1)/2)
            print("We found {0:n} contacting pairs out of {1:n} total pairs of nodes.".format(len(pairs), totalPairs))
            print("(That's {0}%, by the way)".format( round((len(pairs)/(totalPairs))*100,1) ))
        
    
    def filterContacts(self, notSameRes=True, notConsecutiveRes=False, removeIsolatedNodes=True, verbose=0):
        '''Filters network contacts over the system.
        
        The function removes edges and nodes in preparation for network analysis. Traditionally, edges between nodes within the same residue are removed, as well as edges between nodes in consecutive residues within the same polymer chain (protein or nucleic acid). Essentially, nodes that have covalent bonds connecting their node groups can bias the analysis and hide important nonbonded interactions.
        
        The function also removes nodes that are isolated and make no contacts with any other nodes. Examples are ions or solvent residues that were initially included in the system through the preliminary atomated solvent detection routine, but did not reach the contact treshold for being part of the final system.
        
        After filtering nodes and edges, the function updates the MDAnalysis universe and network data.
        
        Args:
            notSameRes (bool) : Remove contacts between nodes in the same residue.
            notConsecutiveRes (bool) : Remove contacts between nodes in consecutive residues.
            removeIsolatedNodes (bool) : Remove nodes with no contacts.
        
        '''
        
        recycleBar = []
        
        for winIndx in tk.log_progress(range(self.numWinds), every=1, size=self.numWinds, name="Window"):
            
            self._filterContactsWindow(self.contactMatAll[winIndx, :, :], 
                                       nodeProgress=recycleBar,
                                       notSameRes=notSameRes, 
                                       notConsecutiveRes=notConsecutiveRes)
    
            #for winIndx in range(self.numWinds):
            print("Window:", winIndx)
            pairs = np.asarray(np.where(np.triu(self.contactMatAll[winIndx, :, :]) > 0)).T
            totalPairs = int(self.contactMat.shape[0]*(self.contactMat.shape[0]-1)/2)
            print("We found {0:n} contacting pairs out of {1:n} total pairs of nodes.".format(len(pairs), totalPairs))
            print("(That's {0}%, by the way)".format( round((len(pairs)/(totalPairs))*100,3) ))
        
        if removeIsolatedNodes:
            
            print("\nRemoving isolated nodes...\n")
            
            # Update unified contact matrix for all windows
            self._genContactMatrix()
            
            # Gets indices for nodes with no contacts
            noContactNodesArray = (np.sum(self.contactMat, axis=1) == 0)
            contactNodesArray   = ~(noContactNodesArray)
            
            # Atom selection for nodes with contact
            contactNodesSel = self.nodesAtmSel.atoms[ contactNodesArray ]
            noContactNodesSel = self.nodesAtmSel.atoms[ noContactNodesArray ]
            
            ### Kepp nodes that belong to residues with at least one network-bound node.
            # This is important in lipids and nucleotides that may have multiple nodes,
            # and only one is connected to the system.
            # First we select *residues* which have at least one node in contact.
            resNoContacts = list(set(noContactNodesSel.residues.ix) - set(contactNodesSel.residues.ix))
            # Then we create an atom selection for residues with no nodes in contact.
            noContactNodesSel = self.nodesAtmSel.residues[resNoContacts]
            # Finaly we update the selection for all nodes that belong to residues with
            # at least one node in contact.
            mask = np.ones(len(self.nodesAtmSel.residues), dtype=bool)
            mask[ resNoContacts ] = False
            contactNodesSel = self.nodesAtmSel.residues[mask].atoms.intersection(self.nodesAtmSel)
            
            # We also have to update the contact matrix that represent nodes which will
            # be kept in the system. For this we will build another mask.
            nodeMask = np.ones(len(self.nodesAtmSel.atoms.ids), dtype=bool)
            for indx,atm in enumerate(self.nodesAtmSel.atoms):
                # We check if the atom belongs to the selection of atoms that will be
                # kept in the system.
                nodeMask[indx] = atm.id in contactNodesSel.atoms.ids
            
            # Checks if there is any node that does not make contacts to ANY other node.
            print("We found {0} nodes with no contacts.".format(len(noContactNodesSel)))
            if verbose:
                for atm in noContactNodesSel.atoms:
                    print(atm)
            
            # Trims matrices
            self.contactMatAll = self.contactMatAll[:, nodeMask,  :]
            self.contactMatAll = self.contactMatAll[:, :,  nodeMask]

            self.contactMat = self.contactMat[ nodeMask, :]
            self.contactMat = self.contactMat[:, nodeMask]
            
            print("\nIsolated nodes removed. We now have {} nodes in the system\n".format(self.contactMatAll[0].shape[0]) )
            print("Running new contact matrix sanity check...")
            
            self.checkContactMat()
            
            #########################
            # Update Universe and network data
            
            print("\nUpdating Universe to reflect new node selection...")
            
            # selStr = "(segid " + " ".join(segIDs) + ") or "
            selStr = " or ".join(["(segid {0} and resid {1})".format(res.segid, res.resid) for res in contactNodesSel.residues])
            
            allSel = self.workU.select_atoms( selStr )
            
            # Merging a selection from the universe returns a new (and smaller) universe
            self.workU = mda.core.universe.Merge(allSel)
            
            # We now create a new universe with coordinates from the selected residues
            self.workU.load_new(mdaAFF(lambda ag: ag.positions.copy(),
                              allSel).run().results, format=mdaMemRead)
            
            # Regenerate selection of atoms that represent nodes.
            # We use the atom selection structure from the previous universe (that still had nodes with
            #   no contacts) to create selection strings and apply them to the new, smaller universe.
            #   This guarantees we have the correct index for all atoms that represent nodes in the new universe.
            selStr = " or ".join([ "(" + tk.getSelFromNode(indx, contactNodesSel, atom=True) + ")" for indx in range(contactNodesSel.n_atoms)])
            
            self.nodesAtmSel = self.workU.select_atoms(selStr)

            self.numNodes = self.nodesAtmSel.n_atoms
            
            # Creates an array relating all atoms in the system to nodes.
            self.atomToNode = np.full(shape=allSel.n_atoms, fill_value=-1, dtype=int)
            
            print("Updating atom-to-node mapping...")
            
            for indx, node in enumerate(tk.log_progress(self.nodesAtmSel.atoms, name="Node")):
                
                # Loops over all atoms in the residue related to the node
                for atm in node.residue.atoms:
                    
                    # Checks if the atom name is listed for the node
                    if atm.name in self.resNodeGroups[node.resname][node.name]:
                        self.atomToNode[atm.ix] = indx
                    
            # Verification: checks if there are any "-1" left. If so, that atom was not assigned a node.
            loneAtms = np.where( self.atomToNode < 0 )[0]
            if len(loneAtms) > 0:
                print("\nERROR: atom assignment incomplete!")
                print("The following atoms were not assigned a node:")
                print( loneAtms )
                print( "Lone atoms (types and residues): ")
                for atm in loneAtms:
                    print( self.workU.atoms[atm] )
            
            #########################
            
    
    def _filterContactsWindow(self, mat, nodeProgress = [], notSameRes=True, notConsecutiveRes=False):
        """
        Filter contacts in a contact matrix.
        
        This function receives a contact matrix and guarantees that there will be no
        self-contacts (a results of some contact detection algorithms).
        Optionally, it can also erase contacts between nodes in the same residue (notSameRes) 
        and between nodes in consecutive residues (notConsecutiveRes).
        """
        
        # Cycles over all nodes in the system. There may be several nodes per residue.
        for node in tk.log_progress(self.nodesAtmSel.atoms, name="Node", userProgress=nodeProgress):

            # Get current node index
            nodeIndx = self.atomToNode[node.ix]
            # Get current node residue
            res = node.residue

            # No contact between the same node (main diagonal) 
            mat[nodeIndx, nodeIndx] = 0

            ## No contact between nodes in same residue
            if notSameRes:
                # Get all node atom(s) in current residue
                resSelection = self.workU.select_atoms(
                    "(segid {0} and resid {1})".format(res.segid, res.resid) \
                    + " and name " + " ".join(self.resNodeGroups[res.resname].keys())
                )
                # Get their node indices
                nodeIndxs = self.atomToNode[resSelection.atoms.ix_array]
                # No correlation between nodes in the same residue.
                # Also zeroes self-correlation.
                for i,j in [(i,j) for i in nodeIndxs for j in nodeIndxs]:
                    mat[i,j] = 0

            # Keeps node from making direct contact to previous residue in chain
            if notConsecutiveRes and (res.resindex -1 >= 0):
                prevRes = self.workU.residues[ res.resindex -1 ]

                if prevRes.segid == res.segid:

                    # Select previous residue in the chain
                    prevResSel = self.workU.select_atoms(
                        "(segid {0} and resid {1})".format(res.segid, prevRes.resid)
                    )
                    # Select the node atom(s) from previous residue
                    prevResSel = prevResSel.select_atoms("name " + \
                                                        " ".join(self.resNodeGroups[prevRes.resname].keys()))

                    # Checks that it is not an ION
                    if prevRes.atoms.n_atoms > 1:
                        # Get the actual node(s) indice(s) from the previous residue
                        nodeIndxs = self.atomToNode[prevResSel.atoms.ix_array]

                        # Zeroes all correlation between nodes in consecutive residues
                        for trgtIndx in nodeIndxs:
                            mat[nodeIndx, trgtIndx] = 0
                            mat[trgtIndx, nodeIndx] = 0

            # Keeps node from making direct contact to following residue in chain
            # (same as above)
            if notConsecutiveRes and (res.resindex +1 < self.workU.residues.n_residues):
                folRes = self.workU.residues[ res.resindex +1 ]

                if folRes.segid == res.segid:

                    folResSel = self.workU.select_atoms(
                        "(segid {0} and resid {1})".format(res.segid, folRes.resid)
                    )
                    folResSel = folResSel.select_atoms("name " + \
                                                    " ".join(self.resNodeGroups[folRes.resname].keys()))

                    if folRes.atoms.n_atoms > 1:
                        nodeIndxs = self.atomToNode[folResSel.atoms.ix_array]

                        for trgtIndx in nodeIndxs:
                            mat[nodeIndx, trgtIndx] = 0
                            mat[trgtIndx, nodeIndx] = 0
    
    def calcCor(self, ncores=1):
        '''Main interface for correlation calculation.
        
        Calculates generalized correlaion coefficients either in serial or in parallel implementations using Python's multiprocessing package. This function wraps the creation of temporary variables in allocates the necessary NumPy arrays for accelerated performance of MDAnalysis algorithms.
        
        .. note:: See also :py:func:`~dynetan.gencor.prepMIc`, :py:func:`~dynetan.gencor.calcMIRnumba2var`, and :py:func:`~dynetan.gencor.calcCorProc`.
        
        Args:
            ncores (int) : Defines how many cores will be used for calculation of generalized correlaion coefficients. Set to `1` in order to use the serial implementation.
        
        '''
        
        if ncores <= 0:
            print("ERROR: number of cores must be at least 1.")
            return 1
        
        # For 3D atom position data
        numDims = 3
        
        print("Calculating correlations...\n")
        
        winLen = int(np.floor(self.workU.trajectory.n_frames/self.numWinds))
        print("Using window length of {} simulation steps.".format(winLen))

        # Allocate the space for all correlations matrices (for all windows).
        self.corrMatAll = np.zeros([self.numWinds, self.numNodes, self.numNodes], dtype=np.float64)
        
        # Stores all data in a dimension-by-frame format.
        traj = np.ndarray( [self.numNodes, numDims, winLen], dtype=np.float64 )
        #traj.nbytes
        
        # Pre-calculate psi values for all frames. (allocation and initialization step)
        psi = np.zeros([winLen+1], dtype=np.float)
        psi[1] = -0.57721566490153

        # Pre-calculate psi values for all frames. (actual calculation step)
        for indx in range(winLen):
            if indx > 0:
                psi[indx + 1] = psi[indx] + 1/(indx)

        # Pre calculates "psi[k] - 1/k" 
        phi = np.ndarray( self.kNeighb+1, dtype=np.float64 )
        for tmpindx in range(1, self.kNeighb+1):
            phi[tmpindx] = psi[tmpindx] - 1/tmpindx
        
        recycleBar = []
        
        if ncores == 1:
            
            print("- > Using single-core implementation.")
            
            for winIndx in tk.log_progress(range(self.numWinds),every=1, size=self.numWinds, name="Window"):
                beg = int(winIndx*winLen)
                end = int((winIndx+1)*winLen)
                
                pairList = np.asarray(np.where(np.triu(self.contactMatAll[winIndx, :, :]) > 0)).T
                
                # Resets the trajectory NP array for the current window.
                traj.fill(0)
                
                # Prepares data for fast calculation of the current window.
                gc.prepMIc(self.workU, traj, beg, end, self.numNodes, numDims)
                
                # Iterates over all pairs of nodes that are in contact.
                for atmList in tk.log_progress(pairList, name="Contact Pair", userProgress=recycleBar ):
                    
                    # Calls the Numba-compiled function.
                    corr = gc.calcMIRnumba2var(traj[atmList, :, :], winLen, numDims, self.kNeighb, psi, phi)
                    
                    # Assures that the Mutual Information estimate is not lower than zero.
                    corr = max(0, corr)
                    
                    # Determine generalized correlation coeff from the Mutual Information
                    if corr:
                        corr = np.sqrt(1-np.exp(-2.0/numDims*corr));
                    
                    self.corrMatAll[winIndx, atmList[0], atmList[1]] = corr
                    self.corrMatAll[winIndx, atmList[1], atmList[0]] = corr
                    
        else:
            
            print("- > Using multi-core implementation with {} threads.".format(ncores))
            
            for winIndx in tk.log_progress(range(self.numWinds),every=1, size=self.numWinds, name="Window"):
                beg = int(winIndx*winLen)
                end = int((winIndx+1)*winLen)
                
                #pairList = np.asarray(np.where(np.triu(contactMatAll[winIndx, :, :]) > 0)).T
                pairList = []
                
                # Build pair list avoiding overlapping nodes (which would require reading the same
                #   trajectory).
                for diag in range(1, self.numNodes):
                    contI = 0
                    contJ = diag
                    while contJ < self.numNodes:
                        if self.contactMatAll[winIndx, contI, contJ]:
                            pairList.append( [contI, contJ] )
                        contI += 1
                        contJ += 1
                
                pairList = np.asarray(pairList)
                
                # Resets the trajectory NP array for the current window.
                traj.fill(0)
                
                # Prepares trajectory data for fast calculation of the current window.
                gc.prepMIc(self.workU, traj, beg, end, self.numNodes, numDims)
                
                # Create queues that feed processes with node pairs, and gather results.
                data_queue = mp.Queue()
                results_queue = mp.Queue()
                for atmList in pairList:
                    data_queue.put(atmList)
                
                # Creates processes.
                procs = []
                for _ in range(ncores):
                    proc = mp.Process(target=gc.calcCorProc, 
                               args=(traj, winLen, psi, phi, numDims, self.kNeighb, data_queue, results_queue)) 
                    proc.start()
                    procs.append(proc)
                
                # Gathers all resuls.
                for _ in tk.log_progress(range(len(pairList)), name="Contact Pair", userProgress=recycleBar ):
                    
                    ## Waits until the next result is available, then puts it in the matrix.
                    result = results_queue.get()
                    
                    self.corrMatAll[winIndx, result[0][0], result[0][1]] = result[1]
                    self.corrMatAll[winIndx, result[0][1], result[0][0]] = result[1]
                    
                # Joins processes.
                for proc in procs:
                    proc.join()
                
        # Sanity Check: Checks that the correlation and contact matrix is symmetric
        for win in range(self.numWinds):
            if not np.allclose(self.corrMatAll[win, :, :], self.corrMatAll[win, :, :].T, atol=0.1):
                print("ERROR: Correlation matrix for window {0} is NOT symmetric!!".format(win))
                
    def calcCartesian(self, backend="serial"):
        '''Main interface for calculation of cartesian distances.
        
        Determines the shortest cartesian distance between atoms in node groups of all network nodes. Using a sampling of simulation frames, the function also calculates statistics on such measures, including mean distance, standard error of the mean, minimum, and maximum.
        This allows analysis comparing network distances and cartesian distances.
        
        .. note:: See also :py:func:`~dynetan.contact.calcDistances` and :py:func:`~dynetan.toolkit.getCartDist`.
        
        Args:
            backend (str) : Defines which MDAnalysis backend will be used for calculation of cartesian distances. Options are `serial` or `openmp`.
        
        '''
        
        ## numFramesDists is used in the calculation of statistics!
        numFramesDists = self.numSampledFrames*self.numWinds
        # numFramesDists = self.numWinds

        selectionAtms = self.workU.select_atoms("all")
        nAtoms = selectionAtms.n_atoms

        nodeDistsTmp = np.zeros( int(self.numNodes*(self.numNodes-1)/2), dtype=np.float64 )

        self.nodeDists = np.zeros( [4, int(self.numNodes*(self.numNodes-1)/2)], dtype=np.float64 )

        print("Sampling a total of {0} frames from {1} windows ({2} per window)...".format(numFramesDists, 
                                                                                        self.numWinds, 
                                                                                        self.numSampledFrames))

        steps = int(np.floor(len(self.workU.trajectory)/numFramesDists))
        maxFrame = numFramesDists*steps

        # Mean distance
        for indx, ts in enumerate(tk.log_progress(self.workU.trajectory[0:maxFrame:steps], 
                                            size=numFramesDists, name="MEAN: Timesteps")):
            
            ct.calcDistances(selectionAtms, self.numNodes, selectionAtms.n_atoms, self.atomToNode, 
                            self.nodeGroupIndicesNP, self.nodeGroupIndicesNPAux, nodeDistsTmp, backend)
            
            # Mean
            self.nodeDists[0, :] += nodeDistsTmp

        self.nodeDists[0, :] /= numFramesDists

        # Initializes the min and max distances with the means.
        self.nodeDists[2, :] = self.nodeDists[0, :]
        self.nodeDists[3, :] = self.nodeDists[0, :]

        ## Standard Error of the Mean
        for indx, ts in enumerate(tk.log_progress(self.workU.trajectory[0:maxFrame:steps], 
                                            size=numFramesDists, name="SEM/MIN/MAX: Timesteps")):
            # serial vs OpenMP
            mdadist.self_distance_array(self.nodesAtmSel.positions, result=nodeDistsTmp, backend=backend)
            
            # Accumulates the squared difference
            self.nodeDists[1, :] += np.square( self.nodeDists[0, :] - nodeDistsTmp )
            
            # Checks for the minimum
            self.nodeDists[2, :] = np.where( nodeDistsTmp < self.nodeDists[2, :], nodeDistsTmp, self.nodeDists[2, :])
            
            # Checks for the maximum
            self.nodeDists[3, :] = np.where( nodeDistsTmp > self.nodeDists[3, :], nodeDistsTmp, self.nodeDists[3, :])
            
        if numFramesDists > 1:
            # Sample standard deviation: SQRT of sum divided by N-1
            self.nodeDists[1, :] = np.sqrt(self.nodeDists[1, :] / (numFramesDists - 1) )
            # SEM:  STD / sqrt(N)
            self.nodeDists[1, :] = self.nodeDists[1, :]/np.sqrt(numFramesDists)

    def calcGraphInfo(self):
        '''Create a graph from the correlation matrix.
        
        Uses NetworkX to create a graph representation of the network. One graph is created per simulation window.
        
        For network analysis, node `distances` are generated with a log transformation of the correlation values. This way, edges between nodes with higher correlation coefficients are considered "closer", with shorter distances, and nodes with low correlation coefficients are "far appart", with larger distance.
        
        .. note:: See also :py:func:`~dynetan.network.calcOptPathPar` and :py:func:`~dynetan.network.calcBetweenPar`.
        
        '''
        self.nxGraphs = []

        for win in range(self.numWinds):
            self.nxGraphs.append( nx.Graph(self.corrMatAll[win, :, :]) )
            
            # We substitute zeros for a non-zero value to avoid "zero division" warnings
            #   from the np.log transformation below.
            self.corrMatAll[ np.where( self.corrMatAll == 0) ] = 10**-11
            
            # Use log transformation for network distance calculations.
            tmpLogTransf = -1.0*np.log(self.corrMatAll[win,:,:])
            
            # Now we guarantee that the previous transformation does not 
            #   create "negative infitite" distances. We set those to zero.
            tmpLogTransf[ np.where( np.isinf(tmpLogTransf) ) ] = 0
            
            # Now we return to zero-correlation we had before.
            self.corrMatAll[ np.where( self.corrMatAll < 10**-10) ] = 0
            
            # Loop over all graph edges and set their distances.
            for pair in self.nxGraphs[win].edges.keys():
                self.nxGraphs[win].edges[(pair[0], pair[1])]["dist"] = tmpLogTransf[pair[0], pair[1]]
            
            # Sets the degree of each node.
            degree_dict = dict(self.nxGraphs[win].degree(self.nxGraphs[win].nodes()))
            nx.set_node_attributes(self.nxGraphs[win], degree_dict, 'degree')
            
    def getDegreeDict(self, window=0):
        '''Compiles a dictionary with node degrees.
        
        This wrapper function uses NetworkX graph object to list the degrees of all nodes.
        
        Args:
            window (int) : Simulation window.
        
        '''
        return dict( self.nxGraphs[window].degree( self.nxGraphs[window].nodes() ) )
        
    def calcOptPaths(self, ncores=1):
        '''Main interface for optimal path calculations.
        
        Calculates optimal paths between all nodes in the network using NetworkX implementation of the Floyd Warshall algorithm. When using more than one core, this function uses Python's `multiprocessing` infrastructure to calculate optimal paths in multiple simulation windows simultaneously.
        
        .. note:: See also :py:func:`~dynetan.network.calcOptPathPar`.
        
        Args:
            ncores (int) : Defines how many cores will be used for calculation of optimal paths. Set to `1` in order to use the serial implementation.
        
        '''
        
        if ncores <= 0:
            print("ERROR: number of cores must be at least 1.")
            return 1
        
        # Sets the network distance array.
        self.distsAll = np.zeros([self.numWinds, self.numNodes, self.numNodes], dtype=np.float)
        
        self.preds = {}
        for i in range(self.numWinds):
            self.preds[i] = 0
        
        if ncores == 1:
            ## Serial Version
            
            for win in tk.log_progress(range(self.numWinds), name="Window"):
                
                ### IMPORTANT!
                # For the FW optimal path determination, we use the "distance" as weight, 
                #  that is, the log-transformation of the correlations. NOT the correlation itself.
                pathsPred, pathsDist = nxFWPD(self.nxGraphs[win], weight='dist')
                
                # Turns dictionary of distances into NumPy 2D array per window
                # Notice the nested list comprehensions.
                self.distsAll[win,:,:] = np.array([[pathsDist[i][j] for i in sorted(pathsDist[j])] for j in sorted(pathsDist)])
                
                # Combines predecessor dictionaries from all windows
                self.preds[win] = copy.deepcopy(pathsPred)
            
        else:
            
            inQueue = mp.Queue()
            outQueue = mp.Queue()

            for win in range(self.numWinds):
                inQueue.put(win)

            # Creates processes.
            procs = []
            for _ in range(ncores):
                procs.append( mp.Process(target=nw.calcOptPathPar, args=(self.nxGraphs, inQueue, outQueue)) )
                procs[-1].start()
                
            for win in tk.log_progress(range(self.numWinds), name="Window"):
                
                ## Waits until the next result is available, then stores it in the object.
                result = outQueue.get()
                
                win = result[0]
                self.distsAll[win,:,:] = np.copy(result[1])
                self.preds[win] = copy.deepcopy(result[2])

            # Joins processes.
            for proc in procs:
                proc.join()
            
        # Get maximum network distance
        self.maxDist = np.max(self.distsAll[self.distsAll != np.inf])

        # Set -1 as distance of nodes with no connecting path (instead of np.inf)
        self.distsAll[ np.where( np.isinf(self.distsAll) ) ] = -1
        
        # Maximum network distance between directly connected nodes (path length == 2)
        # We check connection with the correlation matrix because at times, two nodes may be
        # in contact (physical proximity) but may not have any correlation.
        self.maxDirectDist = max([ self.distsAll[ win, self.corrMatAll[win, :, :] > 0  ].max() for win in range(self.numWinds) ])
        
    def getPath(self, nodeI, nodeJ):
        '''Wrapper for NetworkX reconstruct_path.
        
        The function calls NetworkX's *reconstruct_path* to return the list of nodes that connect `nodeI` to `nodeJ`. This function must only be called **after** a path detection run has been completed (see :py:func:`~dynetan.proctraj.DNAproc.calcOptPaths`).
        
        Args:
            nodeI (int) : Node ID.
            nodeJ (int) : Node ID.
        
        Returns:
            List of node IDs.
        
        '''
        return nx.reconstruct_path(nodeI, nodeJ, self.preds)
        
    def calcBetween(self, ncores=1):
        '''Main interface for betweeness calculations.
        
        Calculates betweeness between all nodes in the network using NetworkX implementation of the betweenness centrality for edges and eigenvector centrality for nodes. When using more than one core, this function uses Python's `multiprocessing` infrastructure to calculate betweeness in multiple simulation windows simultaneously.
        
        .. note:: See also :py:func:`~dynetan.network.calcBetweenPar`.
        
        Args:
            ncores (int) : Defines how many cores will be used for calculation. Set to `1` in order to use the serial implementation.
        
        '''
        
        if ncores <= 0:
            print("ERROR: number of cores must be at least 1.")
            return 1
        
        self.btws = {}
        
        if ncores == 1:
            ## Serial Version
            # Single core version
            for win in tk.log_progress(range(self.numWinds), every=1, size=self.numWinds, name="Window"):
                # Calc all betweeness in entire system.
                ### IMPORTANT!
                # For the betweeness, we only care about the number of shortests paths 
                #   passing through a given edge, so no weight are considered.
                self.btws[win] = nxbetweenness(self.nxGraphs[win], weight=None)
                
                # Creates an ordered dict of pairs with betweenness higher than zero.
                self.btws[win] = {k:self.btws[win][k] for k in self.btws[win].keys() if self.btws[win][k] > 0}
                self.btws[win] = OrderedDict(sorted(self.btws[win].items(), key=lambda t: t[1], reverse=True))
        else:
            
            inQueue = mp.Queue()
            outQueue = mp.Queue()

            for win in range(self.numWinds):
                inQueue.put(win)

            # Creates processes.
            procs = []
            for _ in range(ncores):
                procs.append( mp.Process(target=nw.calcBetweenPar, args=(self.nxGraphs, inQueue, outQueue)) )
                procs[-1].start()
                
            for win in tk.log_progress(range(self.numWinds), name="Window"):
                
                ## Waits until the next result is available, then stores it in the object.
                result = outQueue.get()
                
                win = result[0]
                self.btws[win] = copy.deepcopy(result[1])

            # Joins processes.
            for proc in procs:
                proc.join()
        
    def calcEigenCentral(self):
        '''Wrapper for calculation of node centrality.
        
        Calculates ode centrality for all nodes in all simulation windows. This calculation is relatively inexpensive and is only implemented for serial processing.
        
        All results are stored in the network graph itself.
        
        '''
        for win in range(self.numWinds):
            # Calc all node centrality values in the system.
            cent = nxeigencentrality(self.nxGraphs[win], weight='weight')
            nx.set_node_attributes(self.nxGraphs[win], cent, 'eigenvector')

    def calcCommunities(self):
        '''Calculate node communities using Louvain heuristics.
        
        The function produces sets of nodes that are strongly connected, presenting high correlation coefficients.
        
        It uses Louvain heuristices as an efficient and precise alternative to the classical Girvan–Newman algorithm, which requires much more computing power for large and highly connected networks. This method also maximizes the modularity of the network. It is inherently random, so different calculations performed on the same network data may produce slightly different results.
        
        For more details, see `the original reference <http://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta>`_.
        
        '''
        
        self.nodesComm = {}
        
        for win in range(self.numWinds):
            
            self.nodesComm[win] = {}
            
            communities = community.best_partition(self.nxGraphs[win])

            communitiesLabels = np.unique( np.asarray( list(communities.values()), dtype=int ) )
            
            self.nodesComm[win]["commLabels"] = copy.deepcopy( communitiesLabels )
            
            nx.set_node_attributes(self.nxGraphs[win], communities, 'modularity')
            
            self.nodesComm[win]["commNodes"] = {}
            
            for comm in communitiesLabels:
                # First get a list of just the nodes in that class
                nodesInClass = [n for n in self.nxGraphs[win].nodes() if self.nxGraphs[win].nodes[n]['modularity'] == comm]

                # Then create a dictionary of the eigenvector centralities of those nodes
                nodesInClassEigenVs = {n:self.nxGraphs[win].nodes[n]['eigenvector'] for n in nodesInClass}

                # Then sort that dictionary
                nodesInClassEigenVsOrd = sorted(nodesInClassEigenVs.items(), 
                                                key=itemgetter(1), 
                                                reverse=True)
                nodesInClassEigenVsOrdList = [x[0] for x in nodesInClassEigenVsOrd]
                
                self.nodesComm[win]["commNodes"][comm] = copy.deepcopy( nodesInClassEigenVsOrdList )
            
            # Orders communities based on size.
            communitiesOrdSize = list( sorted(self.nodesComm[win]["commNodes"].keys(),
                                            key=lambda k: len(self.nodesComm[win]["commNodes"][k]),
                                            reverse=True) )
            
            self.nodesComm[win]["commOrderSize"] = copy.deepcopy( communitiesOrdSize )
            
            # Orders communities based on highest eigenvector centrality of all its nodes.
            communitiesOrdEigen = list( sorted(self.nodesComm[win]["commNodes"].keys(), 
                                            key=lambda k: self.nxGraphs[win].nodes[self.nodesComm[win]["commNodes"][k][0]]['eigenvector'], 
                                            reverse=True) )
            
            self.nodesComm[win]["commOrderEigenCentr"] = copy.deepcopy( communitiesOrdEigen )
    
    def interfaceAnalysis(self, selAstr, selBstr, betweenDist = 15, samples = 10):
        '''Detects interface between molecules.
        
        Based on user-defined atom selections, the function detects residues (and their network nodes) that are close to the interface between both atom selections. That may include amino acids in the interface, as well as ligands, waters and ions.
        
        Only nodes that have edges to nodes on the side of the interface are selected.
        
        Using a sampling of simulation frames assures that transient contacts will be detected by this analysis.
        
        Args:
            selAstr (str) : Atom selection.
            selBstr (str) : Atom selection.
            betweenDist (float) : Cutoff distance for selection of atoms that are within *betweenDist* from both selections.
            samples (int) : Number of frames to be sampled for detection of interface residues.
        
        '''
        
        # Select the necessary stride so that we get *samples*.
        stride = int(np.floor(len(self.workU.trajectory)/samples))
        
        selPtn = self.workU.select_atoms(selAstr)
        selNcl = self.workU.select_atoms(selBstr)

        contactNodes = set()

        # Find selection of atoms that are within "betweenDist" from both selections.
        # Get selection of nodes represented by the atoms by sampling several frames.
        for ts in tk.log_progress(self.workU.trajectory[:samples*stride:stride], every=1, 
                            name="Samples",size=samples):
            
            contactSel = mdaB(self.workU.select_atoms("all"), selPtn, selNcl, betweenDist )    
            contactNodes.update(np.unique( self.atomToNode[ contactSel.atoms.ix_array ] ))
        
        # Makes it into a list for better processing
        contactNodesL = np.asarray(list(contactNodes))
        
        # Sanity check.
        # Verifies possible references from atoms that had no nodes.
        if len(contactNodesL[ contactNodesL < 0 ]):
            print("ERROR! There are {} atoms not represented by nodes! Verify your universe and atom selection.".format(len(contactNodesL[ contactNodesL < 0 ])))
        
        # These are all nodes in both selections.
        numContactNodesL = len(contactNodes)
        
        #print("{0} nodes found in the interface.".format(numContactNodesL))

        # Filter pairs of nodes that have contacts
        contactNodePairs = []
        for i in range(numContactNodesL):
            for j in range(i,numContactNodesL):
                nodeI = contactNodesL[i]
                nodeJ = contactNodesL[j]
                if max([ self.corrMatAll[win, nodeI, nodeJ] for win in range(self.numWinds) ]) > 0:
                    contactNodePairs.append( (nodeI, nodeJ) )

        # These are all pairs of nodes that make direct connections. These pairs WILL INCLUDE
        #    pairs where both nodes are on the same side of the interface.
        contactNodePairs = np.asarray( contactNodePairs, dtype=np.int )
        
        #print("{0} contacting node pairs found in the interface.".format(len(contactNodePairs)))

        def inInterface(nodesAtmSel, i, j):
            segID1 = nodesAtmSel.atoms[i].segid
            segID2 = nodesAtmSel.atoms[j].segid
            
            if (segID1 != segID2) and ((segID1 in self.segIDs) or (segID2 in self.segIDs)):
                return True
            else:
                return False
        
        # These are pairs where the nodes are NOT on the same selection, that is, pairs that connect
        #   the two atom selections.
        self.interNodePairs = [ (i,j) for i,j in contactNodePairs if inInterface(self.nodesAtmSel, i, j) ]
        self.interNodePairs = np.asarray( self.interNodePairs, dtype=np.int )
        print("{0} pairs of nodes connecting the two selections.".format(len(self.interNodePairs)))

        self.contactNodesInter = np.unique(self.interNodePairs)
        print("{0} unique nodes in interface node pairs.".format(len(self.contactNodesInter)))
        
        
        
        
        
        
        
        
