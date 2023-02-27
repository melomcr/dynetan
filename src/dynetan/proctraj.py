#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: melomcr

from . import gencor as gc
from . import contact as ct
from . import network as nw
from . import datastorage as ds

import numpy as np
import numpy.typing as npt

import MDAnalysis as mda

import multiprocessing as mp

import networkx as nx

import community

from networkx import eigenvector_centrality_numpy as nxeigencentrality
from networkx import edge_betweenness_centrality as nxbetweenness
from networkx.algorithms.shortest_paths.dense import \
    floyd_warshall_predecessor_and_distance as nxFWPD

from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch as mdaANS
from MDAnalysis.coordinates.memory import MemoryReader as mdaMemRead
from MDAnalysis.analysis.base import AnalysisFromFunction as mdaAFF
from MDAnalysis.analysis.distances import between as mdaB

from colorama import Fore
from operator import itemgetter
from collections import OrderedDict
from collections import defaultdict
import queue  # So that we can catch the exceptions for empty queues

import copy

# For timing and benchmarks
from timeit import default_timer as timer
from datetime import timedelta

from typing import Literal, Union, Any, Tuple

dist_modes_literal = Literal["all", "capped"]
dist_modes = ["all", "capped"]

backend_types_literal = Literal["serial", "openmp"]
backend_types = ["serial", "openmp"]

##################################################
##################################################


def is_proteic(resname: str) -> bool:
    mda_known_prot_res = mda.core.selection.ProteinSelection.prot_res
    return resname in mda_known_prot_res


def in_interface(nodes_atm_sel: mda.AtomGroup,
                 node_i: int,
                 node_j: int,
                 seg_ids: list) -> bool:
    segID1 = nodes_atm_sel.atoms[node_i].segid
    segID2 = nodes_atm_sel.atoms[node_j].segid

    if (segID1 != segID2) and (
            (segID1 in seg_ids) or (segID2 in seg_ids)):
        return True
    else:
        return False


class DNAproc:
    """The Dynamic Network Analysis processing class contains the
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

    """

    def __init__(self, notebookMode=True):  # type: ignore  # Just during refactoring
        """Constructor

        Sets initial values for class variables.

        """

        self.dnaData: ds.DNAdata = ds.DNAdata()

        # Basic DNA parameters
        self.contactPersistence = 0.75
        self.cutoffDist = 4.5
        self.numSampledFrames = 1
        self.numWinds = 1

        # Number of neighbours for Generalized Correlation estimate
        self.kNeighb = 7

        self.solventNames: list[str] = []
        self.segIDs: list[str] = []

        self.usrNodeGroups: dict[str, dict[str, set[str]]] = {}
        self.resNodeGroups: dict[str, dict[str, set[str]]] = {}

        self.allResNamesSet = None
        self.selResNamesSet = None
        self.notSelResNamesSet = None
        self.notSelSegidSet = None

        self.workU: mda.Universe = None

        self.nodesAtmSel: mda.core.groups.AtomGroup = None
        self.atomToNode = None

        self.numNodes = None

        self.nodeGroupIndicesNP = None
        self.nodeGroupIndicesNPAux = None
        self.contactMatAll = None
        self.contactMat = None
        self.corrMatAll = None

        self.nodeDists = None
        self.nxGraphs = None
        self.distsAll = None
        self.preds = None
        self.maxDist = None
        self.maxDirectDist = None
        self.btws = None
        self.nodesComm = None

        self.interNodePairs = None
        self.contactNodesInter = None

        self.distanceMode: int = ct.MODE_ALL

        self.notebookMode: bool = notebookMode

        if self.notebookMode:
            from tqdm.notebook import tqdm_notebook as tqdm
            self.asciiMode = False

        else:
            from tqdm import tqdm
            self.asciiMode = True

        self.progBar = tqdm

    def setNumWinds(self, num_winds: int = 1) -> None:
        """Set number of windows.

        This will determine the number of windows into which the
        trajectory will be split.

        Usage tip: If there are several concatenated replicas of the same system,
        make sure all have the same number of frames so that the split will
        extract each replica in a different window.

        Args:
            num_winds (int) : Number of windows.
        """

        assert isinstance(num_winds, int), "Wrong argument type!"
        assert num_winds > 0, "We need at least one window!"
        self.numWinds = num_winds

    def setNumSampledFrames(self, n_smpld_frms: int = 1) -> None:
        """Set number of frames to be sampled for solvent detection.

        This will determine how many frames will be sampled for solvent detection
        per window, and for estimation of cartesian distance between node groups.

        Args:
            n_smpld_frms (int) : Number of sampled frames per window.
        """

        assert isinstance(n_smpld_frms, int), "Wrong argument type!"
        assert n_smpld_frms > 0, "We need at least one frame!"
        self.numSampledFrames = n_smpld_frms

    def setCutoffDist(self, cutoff_dist: float = 4.5) -> None:
        """Set cartesian distance cutoff for contact detection.

        For all atom simulations, assuming only heavy atoms (non-hydrogen atoms)
        were kept in the system, this number is usually set to 4.5 Angstroms.

        Args:
            cutoff_dist (float) : Cutoff distance for contact detection.

        """

        assert isinstance(cutoff_dist, (int, float)), "Wrong argument type!"
        assert cutoff_dist > 0, "We need at least one frame!"
        self.cutoffDist = float(cutoff_dist)

    def setContactPersistence(self, contact_persistence: float = 0.75) -> None:
        """Set contact persistence cutoff for contact detection.

        Args:
            contact_persistence (float) : Ratio of total trajectory frames
                needed to consider a pair of nodes to be in contact.
                Usually set to 0.75 (75% of total trajectory).

        """

        assert isinstance(contact_persistence, float), "Wrong argument type!"
        assert isinstance(contact_persistence, float), "Wrong argument type!"
        assert contact_persistence > 0.0, "Persistence needs to be in (0,1] interval!"
        assert contact_persistence < 1.0, "Persistence needs to be in (0,1] interval!"
        self.contactPersistence = contact_persistence

    def setSolvNames(self, solvent_names: list[str]) -> None:
        """Set name of solvent molecule residue.

        Args:
            solvent_names (list) : List of residue names used as solvent.

        """

        assert isinstance(solvent_names, list), "Wrong argument type!"
        for solv in solvent_names:
            assert isinstance(solv, str), "Wrong argument type!"
        self.solventNames = solvent_names

    def seth2oName(self, solvent_names: list[str]) -> None:
        """Set name of solvent molecule residue.

        Args:
            solvent_names (list) : List of residue names used as solvent.

        """
        print("WARNING: This method will be deprecated. "
              "The method setSolvNames will replace it.")
        self.setSolvNames(solvent_names)

    def setSegIDs(self, seg_ids: list[str]) -> None:
        """Set segment IDs for biomolecules ot be analyzed.

        Args:
            seg_ids (list) : List of Segment IDs to be included in network analysis.

        """

        assert isinstance(seg_ids, list), "Wrong argument type!"
        for idstr in seg_ids:
            assert isinstance(idstr, str), "Wrong argument type!"
        self.segIDs = seg_ids

    def setCustomResNodes(self, customResNodes: Any) -> None:
        """Set atoms that will represent nodes in user defined residues.

        .. note:: THIS METHOD HAS BEEN DEPRECATED. It has been fully replaced
            by :py:func:`setNodeGroups`.

        Args:
            customResNodes (dict) : Dictionary mapping residue names with
                lists of atom names that will represent network nodes.

        """
        print("### The method `setCustomResNodes` is now deprecated and "
              "replaced with the `setNodeGroups` method.")
        raise DeprecationWarning

    def setNodeGroups(self, node_groups: dict[str, dict[str, set[str]]]) -> None:
        """Set atoms that will represent node groups in user-defined residues.

        Network Analysis will create one network node per standard amino acid
        residue (in the alpha carbon). For other residues, the user must specify
        atom(s) that will represent a node.
        This function is used to define the heavy atoms that compose each node
        group for user-defined nodes.

        Args:
            node_groups (dict) : Nested dictionary mapping residue names
                with atom names that will represent network nodes, and sets of
                heavy atoms used to define node groups.

        """

        assert isinstance(node_groups, dict), "Wrong argument type!"
        for resstr, nodedict in node_groups.items():
            assert isinstance(resstr, str), "Wrong argument type!"
            assert isinstance(nodedict, dict), "Wrong argument type!"

            for nodestr, nodeatms in nodedict.items():
                assert isinstance(nodestr, str), "Wrong argument type!"
                assert isinstance(nodeatms, set), "Wrong argument type!"

                for atmstr in nodeatms:
                    assert isinstance(atmstr, str), "Wrong argument type!"

        self.usrNodeGroups = node_groups

    def setUsrNodeGroups(self, node_groups: dict[str, dict[str, set[str]]]) -> None:
        print("WARNING: This method will be deprecated. "
              "The method setNodeGroups will replace it.")

        self.setNodeGroups(node_groups)
        raise DeprecationWarning

    def setDistanceMode(self, mode: dist_modes_literal = "all") -> None:
        """Set the distance calculation method to find nodes in contact.

        The supported options are:

        1. ``all``, which calculates all-to-all distances between selected atoms
           in the system.

        2. ``capped``, which uses a kdtree algorithm to only calculate distances
           between atoms closer than the network distance cutoff.

        The "all" option will be faster for smaller systems. The "capped" option
        will benefit larger systems as it will require less memory.

        .. note:: See also :py:func:`setCutoffDist`.

        Args:
            mode (str) : Distance calculation mode.

        """

        assert mode in dist_modes, f"Only allowed modes are {dist_modes}"

        if mode == "all":
            self.distanceMode = ct.MODE_ALL
        elif mode == "capped":
            self.distanceMode = ct.MODE_CAPPED

    def getU(self) -> Any:
        """Return MDAnalysis universe object.
        """
        return self.workU

    def saveData(self, file_name_root: str = "dnaData") -> None:
        """Save all network analysis data to file.

        This function automates the creation of a
        :py:class:`~dynetan.datastorage.DNAdata` object, the placement of data
        in the object, and the call to its
        :py:func:`~dynetan.datastorage.DNAdata.saveToFile` function.

        Args:
            file_name_root (str) : Root of the multiple data files to be writen.

        """

        self.dnaData = ds.DNAdata()

        self.dnaData.nodesIxArray = self.nodesAtmSel.ix_array
        self.dnaData.numNodes = self.numNodes
        self.dnaData.atomToNode = self.atomToNode
        self.dnaData.contactMat = self.contactMat

        self.dnaData.corrMatAll = self.corrMatAll

        # Cartesian distances between nodes.
        if self.nodeDists is not None:
            self.dnaData.nodeDists = self.nodeDists

        self.dnaData.distsAll = self.distsAll
        self.dnaData.preds = self.preds
        self.dnaData.maxDist = self.maxDist
        self.dnaData.maxDirectDist = self.maxDirectDist
        self.dnaData.btws = self.btws
        self.dnaData.nodesComm = self.nodesComm
        self.dnaData.nxGraphs = self.nxGraphs

        if self.interNodePairs is not None:
            self.dnaData.interNodePairs = self.interNodePairs

        if self.contactNodesInter is not None:
            self.dnaData.contactNodesInter = self.contactNodesInter

        self.dnaData.saveToFile(file_name_root)

    def saveReducedTraj(self, file_name_root: str = "dnaData", stride: int = 1) -> None:
        """Save a reduced trajectory to file.

        This function automates the creation of a reduced DCD trajectory file
        keeping only the atoms used for Dynamical Network Analysis. It also
        creates a matching PDB file to maintain atom and residue names.

        Args:
            file_name_root (str) : Root of the trajectory and structure
                files to be writen.
            stride (int) : Stride used to write the trajectory file.

        """

        dcdVizFile = file_name_root + "_reducedTraj.dcd"

        totalFrames: int = int(len(self.workU.trajectory[::stride]))

        with mda.Writer(dcdVizFile, self.workU.atoms.n_atoms) as W:
            for _ in self.progBar(self.workU.trajectory[::stride], desc="Frames",
                                  total=totalFrames, ascii=self.asciiMode):
                W.write(self.workU.atoms)

        pdbVizFile = file_name_root + "_reducedTraj.pdb"

        with mda.Writer(pdbVizFile,
                        multiframe=False,
                        bonds="conect",
                        n_atoms=self.workU.atoms.n_atoms) as PDB:
            PDB.write(self.workU.atoms)

    def loadSystem(self, str_fn: str, traj_fns: Union[str, list[str]]) -> None:
        """
        Loads Structure and Trajectory files to an MDAnalysis universe.

        Args:
            str_fn (str) : Path to structure file, such as a PSF, PDB,
                Gro, or other file formats accepted by MDAnalysis.
            traj_fns (str | List(str) ) : Path to one or more trajectory
                files. MDAnalysis will automatically concatenate trajectories if
                multiple files are passed.

        """

        self.workU = mda.Universe(str_fn, traj_fns)

    def checkSystem(self) -> None:
        """Performs a series of sanity checks.

        This function checks if the user-defined data and loaded simulation data
        are complete and compatible. This will print a series of diagnostic
        messages that should be used to verify if all calculations are set up
        as desired.

        """

        assert self.workU is not None, "ERROR! This function can only be called " \
                                       "after loading a system. Check your universe!"

        # Initialize residue name sets for system selection
        allResNamesSet = set()
        selResNamesSet = set()
        notSelSegidSet = set()

        print(Fore.BLUE + "Residue verification:\n" + Fore.RESET)

        # Loop over segments and checks residue names
        for segment in self.workU.segments:
            segid = segment.segid

            resNames = set([res.resname for res in segment.residues])

            if segid in self.segIDs:
                print("---> SegID ",
                      Fore.GREEN + segid,
                      Fore.RESET + ":",
                      len(resNames),
                      "unique residue types:")
                print(resNames)
                print()
                selResNamesSet.update(resNames)
            else:
                notSelSegidSet.add(segid)

            allResNamesSet.update(resNames)

        print("---> {0} total selected residue types:".format(len(selResNamesSet)))
        print(selResNamesSet)
        print()

        notSelResNamesSet = allResNamesSet - selResNamesSet

        print(("---> {0} " + Fore.RED + "not-selected" + Fore.RESET +
               " residue types in other segments:").format(len(notSelResNamesSet)))
        print(notSelResNamesSet)
        print()

        print("---> {0} total residue types:".format(len(allResNamesSet)))
        print(allResNamesSet)
        print()

        selRes = self.workU.select_atoms("segid " + " ".join(self.segIDs))
        print("---> " + Fore.GREEN +
              "{0} total residues".format(len(selRes.residues))
              + Fore.RESET + " were selected for network analysis.")
        print()

        print(Fore.BLUE + "Segments verification:\n" + Fore.RESET)

        print(("---> {0} " + Fore.GREEN + "selected" + Fore.RESET +
               " segments:").format(len(self.segIDs)))
        print(self.segIDs)
        print()

        print(("---> {0} " + Fore.RED + "not-selected" + Fore.RESET +
               " segments:").format(len(notSelSegidSet)))
        print(sorted(notSelSegidSet, key=str.lower))
        print()

        self.allResNamesSet = allResNamesSet
        self.selResNamesSet = selResNamesSet
        self.notSelResNamesSet = notSelResNamesSet
        self.notSelSegidSet = notSelSegidSet

    def selectSystem(self,
                     withSolvent: bool = False,
                     inputSelStr: str = "",
                     verbose: int = 0) -> None:
        """Selects all atoms used to define node groups.

        Creates a final selection of atoms based on the user-defined residues and
        node groups. This function also automates solvent and ion detection, for
        residues that make significant contacts with network nodes. Examples are
        structural water molecules and ions.

        This function will automatically remove all hydrogen atoms from the system,
        since they are not used to detect contacts or to calculate correlations.
        The standard selection string used is "not (name H* or name [123]H*)"

        Ultimately, an MDAnalysis universe is created with the necessary simulation
        data, reducing the amount of memory used by subsequent analysis.

        Args:

            withSolvent (bool): Controls if the function will try to automatically
                detect solvent molecules.

            inputSelStr (str): Uses a user-defined selection for the system. This
                disables automatic detection of solvent/ions/lipids and other
                residues that may have transient contact with the target system.

            verbose (int): Controls the verbosity of output.

        """

        assert isinstance(withSolvent, bool)
        assert isinstance(inputSelStr, str)
        assert isinstance(verbose, int)

        if self.notSelSegidSet is None:
            print("Checking system for information on residues and segments...")
            self.checkSystem()

        if (not withSolvent) and (not self.solventNames):
            err_str = "ERROR: Automatic removal of all solvent molecules can " \
                      "only happen if we have a list of solvent residue names, " \
                      "but no solvent names were provided. Aborting function call."
            raise Exception(err_str)

        if inputSelStr.strip():
            print("Using user-defined selection string:")
            print(inputSelStr)
            print("\nATTENTION: automatic identification of "
                  "solvent and ions is DISABLED.")

            initialSel = self.workU.select_atoms(inputSelStr)

        else:
            if withSolvent:
                # For automatic solvent detection, we use the segIDs that were
                # not selected by the user as targets for network analysis.
                if self.notSelSegidSet:
                    selStr = "(not (name H* or name [123]H*)) "
                    selStr += "and segid " + " ".join(self.notSelSegidSet)
                    checkSet = self.workU.select_atoms(selStr)
                else:
                    print("WARNING: All segments have been selected for Network "
                          "Analysis, none are left for automatic identification "
                          "of structural solvent molecules or lipids.")
                    checkSet = None
            else:
                # Without solvent detection, the new selection removes all solvent
                # molecules from the selection but keeps other segments that may
                # ions or ligands that may be structural and necessary for the
                # analysis.
                if self.notSelSegidSet:
                    selStr = "segid " + " ".join(self.notSelSegidSet)
                    selStr += " and not resname " + " ".join(self.solventNames)
                    selStr += " and not (name H* or name [123]H*)"
                    checkSet = self.workU.select_atoms(selStr)
                else:
                    print("WARNING: All segments have been selected for Network "
                          "Analysis, none are left for automatic identification "
                          "of transient contacts.")
                    checkSet = None

            if checkSet:
                numAutoFrames: int = self.numSampledFrames * self.numWinds

                stride = int(np.floor(len(self.workU.trajectory) / numAutoFrames))

                print("Checking {0} frames (striding {1})...".format(
                    numAutoFrames, stride))

                selRes = self.workU.select_atoms("segid " + " ".join(self.segIDs))
                searchSelRes = selRes.select_atoms("not (name H* or name [123]H*)")

                # Keeps a set with all residues that were close to the interaction
                #  region in ALL sampled timesteps

                resIndexDict: dict[int, int] = defaultdict(int)
                for ts in self.progBar(
                        self.workU.trajectory[:numAutoFrames * stride:stride],
                        desc="Frames", total=numAutoFrames, ascii=self.asciiMode):

                    # Creates neighbor search object. We pass the atoms we want to check,
                    #   and then search using the main selection.
                    # This is expensive because it creates a KD-tree for every frame,
                    #   but the search for Neighbors is VERY fast.
                    searchNeigh = mdaANS(checkSet)

                    resNeigh = searchNeigh.search(searchSelRes, self.cutoffDist)

                    for indx in resNeigh.residues.ix:
                        resIndexDict[indx] += 1

                resIndxList = [k for k, v in resIndexDict.items()
                               if v > int(numAutoFrames * self.contactPersistence)]

                checkSetMin = self.workU.residues[np.asarray(resIndxList, dtype=int)]

                newResStr = "{} extra residues will be added to the system."
                print(newResStr.format(len(checkSetMin.resnames)))

                if verbose > 0:
                    print("New residue types included in the system selection:")
                    for resname in set(checkSetMin.resnames):
                        print(resname)

                    if verbose > 1:
                        print("New residues included in the system selection:")
                        for res in set(checkSetMin.residues):
                            print(res)

                selStr = "segid " + " ".join(self.segIDs)
                initialSel = self.workU.select_atoms(selStr)
                initialSel = initialSel.union(checkSetMin.atoms)
                initialSel = initialSel.select_atoms("not (name H* or name [123]H*)")

            else:
                # In case we do not have any residues in other segments to
                # check for contacts, we take all user-selected segments and
                # create the system for analysis.
                selStr = "segid " + " ".join(self.segIDs)
                initialSel = self.workU.select_atoms(selStr)
                initialSel = initialSel.select_atoms("not (name H* or name [123]H*)")

        print("The initial universe had {} atoms.".format(len(self.workU.atoms)))

        # Merging a selection from the universe returns a new (and smaller) universe
        self.workU = mda.core.universe.Merge(initialSel)

        print("The final universe has {} atoms.".format(len(self.workU.atoms)))

        # We now load the new universe to memory, with coordinates
        # from the selected residues.

        print("Loading universe to memory...")

        resObj = mdaAFF(lambda ag: ag.positions.copy(), initialSel).run().results

        # This checks the type of the MDAnalysis results. Prior to version 2.0.0,
        # MDA returned a numpy.ndarray with the trajectory coordinates. After
        # version 2.0.0, it returns a results object that contains the trajectory.
        # With this check, the code can handle both APIs.
        if not isinstance(resObj, np.ndarray):
            resObj = resObj['timeseries']

        self.workU.load_new(resObj, format=mdaMemRead)

    def prep_node_groups(self, autocomp_groups: bool = True) -> None:
        """Prepare node groups and check system for unknown residues

        This function will load the user-defined node groups into this object
        and will create node groups from standard proteic residues and trivial
        single-atom residues such as ions.

        Args:
            autocomp_groups (bool): Method will automatically add atoms from residues
                with defined node groups, as long as the atom is bound to another
                atom included in a node group. This is intended to facilitate the
                inclusion of hydrogen atoms to node groups without hard coded user
                definitions.

        Returns:
            none

        """

        assert isinstance(autocomp_groups, bool)

        # Initialize the global node group dictionary
        self.resNodeGroups = {}

        # Include user-defined node groups
        self.resNodeGroups.update(self.usrNodeGroups)

        # Iterate over all residues in the system to check node groups
        for res in self.workU.residues:

            if res.resname in self.resNodeGroups.keys():

                if is_proteic(res.resname) and autocomp_groups:
                    # Update known protein residues to include hydrogen atoms
                    self.resNodeGroups[res.resname]["CA"].update(set(res.atoms.names))
                elif autocomp_groups:
                    # Adds hydrogen atoms to a groups of atoms in every residue.
                    for atm in res.atoms:
                        # Assume it is a hydrogen and bind it to the group of the atom
                        # it is connected to.
                        atmSet = set.union(*self.resNodeGroups[res.resname].values())
                        if atm.name not in atmSet:
                            boundTo = atm.bonded_atoms[0].name

                            # If there are multiple sub-groups in the same residue
                            # the atom will be added to the same group that its
                            # bound atom belongs to.
                            for key, val in self.resNodeGroups[res.resname].items():
                                if boundTo in val:
                                    self.resNodeGroups[res.resname][key].add(atm.name)
            else:
                # Verifies if there are unknown residues

                if is_proteic(res.resname):
                    # Automatically create node groups for standard protein residues.
                    # This will include hydrogen atoms, if they are still in the system.
                    self.resNodeGroups[res.resname] = {}
                    self.resNodeGroups[res.resname]["CA"] = set(res.atoms.names)

                elif len(res.atoms) == 1:
                    # For residues that are not proteic, and that have one atom (IONS)
                    # Creates the "node group" and the atom name for the node.
                    self.resNodeGroups[res.resname] = {}
                    resAtmName = res.atoms.names[0]
                    setAtmNames = set(res.atoms.names)
                    self.resNodeGroups[res.resname][resAtmName] = setAtmNames

                elif len(res.atoms) > 1:
                    print((Fore.RED + "Unknown residue type" + Fore.RESET +
                           " {0}, from segment {1}").format(res.resname, res.segid))

                    error_str = f"Residue {res.resname} does not have a defined " \
                                f"node group!"
                    raise Exception(error_str)

            # Sanity check
            # Verify if there are node atoms in the system that are NOT
            # accounted for in node groups.
            setResNodes = set(self.resNodeGroups[res.resname].keys())
            kMissing = setResNodes.difference(set(res.atoms.names))
            if kMissing:
                errorStr = (Fore.RED + "ERROR!" + Fore.RESET +
                            " residue {0} segid {1} resid {2} does not contain "
                            "all node atoms. Missing atoms: {3}"
                            ).format(res.resname,
                                     res.segid,
                                     res.resid,
                                     " ".join(kMissing))
                print(errorStr)

                raise Exception(errorStr)

    def _check_atom_to_node_mapping(self, verbose: int = 1) -> None:
        # Verification: checks if there are any "-1" left.
        # If so, that atom was not assigned a node.
        loneAtms = np.where(self.atomToNode < 0)[0]
        if len(loneAtms) > 0:
            if verbose:
                print("ERROR: Atoms were not assigned to any node!")
                print("This can be a problem with your definition of "
                      "nodes and atom groups.")
                print("Lone atoms: ")
                print(loneAtms)
                print("Lone atoms (types and residues): ")
                if verbose > 1:
                    for atm in loneAtms:
                        print(self.workU.atoms[atm])

            raise Exception("Found atoms not assigned to any node group")

    def prepareNetwork(self,
                       verbose: int = 0,
                       autocomp_groups: bool = True) -> None:
        """Prepare network representation of the system.

        Checks if we know how to treat all types of residues in the final system
        selection. Every residue will generate one or more nodes in the final
        network. This function also processes and stores the groups of atoms
        that define each node group in specialized data structures.

        .. note:: We need this special treatment because the residue information
            in the topology file may list atoms in an order that separates atoms
            from the same node group. Even though atoms belonging to the same residue
            are contiguous, atoms in our arbitrary node groups need not be contiguous.
            Since amino acids have just one node, they will have just one range of
            atoms but nucleotides and other residues may be different.

        """

        from itertools import groupby
        from itertools import chain
        from operator import itemgetter

        assert isinstance(autocomp_groups, bool)
        assert isinstance(verbose, int)

        self.prep_node_groups(autocomp_groups)

        # The following steps will create an atom selection object
        # for atoms that represent nodes.

        # Builds list of selection statements
        selStrL = ["(resname {0} and name {1})".format(k, " ".join(v.keys()))
                   for k, v in self.resNodeGroups.items()]

        # Combines all statements into one selection string
        selStr = " or ".join(selStrL)

        if verbose:
            print("Selection string for atoms that represent network nodes:")
            print(selStr)

        self.nodesAtmSel = self.workU.select_atoms(selStr)

        self.numNodes = self.nodesAtmSel.n_atoms

        if verbose:
            print("Preparing nodes...")

        self.atomToNode = np.full(shape=len(self.workU.atoms),
                                  fill_value=-1,
                                  dtype=int)

        # Creates an array relating atoms to nodes.
        for indx, node in enumerate(self.progBar(self.nodesAtmSel.atoms,
                                                 desc="Nodes",
                                                 ascii=self.asciiMode)):

            # Loops over all atoms in the residue related to the node
            for atm in node.residue.atoms:

                # Checks if the atom name is listed for the node
                if atm.name in self.resNodeGroups[node.resname][node.name]:
                    self.atomToNode[atm.ix] = indx

        self._check_atom_to_node_mapping(verbose)

        # Determine groups of atoms that define each node.
        # We need all this because the topology in the PSF may
        #   list atoms in an order that separates atoms from the same node.
        #   Even though atoms in the same *residue* are contiguous,
        #   atoms in our arbitrary node groups need not be contiguous.
        # Since amino acids have just one node, they will have just one range
        #   but nucleotides and other residues may be different.

        nodeGroupRanges = {}
        nodeGroupIndicesL = []

        for x in np.unique(self.atomToNode):
            data = np.where(self.atomToNode == x)[0]

            ranges = []
            for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                # Creates an iterable from the group object.
                groupMap = (map(itemgetter(1), g))
                # Creates a list using the iterable
                group = list(map(int, groupMap))
                # Appends to the list of ranges the first and last items in each range.
                ranges.append((group[0], group[-1]))

            nodeGroupRanges[x] = ranges

            # Transforms the ranges into lists
            tmp = [[x for x in range(rang[0], rang[1] + 1)]
                   for rang in nodeGroupRanges[x]]

            # Combine lists into one list
            nodeGroupIndicesL.append(list(chain.from_iterable(tmp)))

        nodeGroupIndices: npt.NDArray = np.asarray(nodeGroupIndicesL,
                                                   dtype=object)

        self.nodeGroupIndicesNP = np.asarray(
            list(chain.from_iterable(nodeGroupIndices)),
            dtype=int)

        self.nodeGroupIndicesNPAux = np.zeros(self.numNodes, dtype=int)

        for indx, atmGrp in enumerate(nodeGroupIndices[1:], start=1):
            self.nodeGroupIndicesNPAux[indx] = len(nodeGroupIndices[indx - 1]) + \
                                               self.nodeGroupIndicesNPAux[indx - 1]

        if verbose:
            print("Nodes are ready for network analysis.")

    def alignTraj(self,
                  selectStr: str = "",
                  inMemory: bool = True,
                  verbose: int = 0) -> None:
        """
        Wrapper function for MDAnalysis trajectory alignment tool.

        Args:
            inMemory: Controls if MDAnalysis `AlignTraj` will run in memory.
            selectStr: User defined selection for alignment. If empty, will
                use default: Select all user-defined segments and exclude
                hydrogen atoms.
            verbose: Controls verbosity level.

        Returns:
            none
        """

        assert isinstance(inMemory, bool)
        assert isinstance(selectStr, str)
        assert isinstance(verbose, int)

        from MDAnalysis.analysis import align as mdaAlign

        if not selectStr.strip():
            selectStr = "segid " + " ".join(self.segIDs) + \
                         " and not (name H* or name [123]H*)"

        if verbose > 1:
            print("Using alignment selection string: ")
            print(selectStr)

        # Set the first frame as reference for alignment.
        # The opperator[] is used in MDanalysis to select a reference frame for
        # alignment.
        _ = self.workU.trajectory[0]

        alignment = mdaAlign.AlignTraj(self.workU, self.workU,
                                       select=selectStr,
                                       verbose=verbose,
                                       in_memory=inMemory,
                                       weights="mass")
        alignment.run()

    def _contactTraj(self,
                     contact_mat: np.ndarray,
                     beg: int = 0,
                     end: int = -1,
                     stride: int = 1,
                     verbose: int = 0) -> None:
        """Wrapper for contact calculation per trajectory window.

        Pre allocates the necessary temporary NumPy arrays to speed up calculations.

        """

        assert isinstance(contact_mat, np.ndarray)
        assert isinstance(beg, int)
        assert isinstance(end, int)
        assert isinstance(stride, int)
        assert isinstance(verbose, int)

        # Creates atom selection for distance calculation
        selectionAtms = self.workU.select_atoms("all")

        nAtoms = selectionAtms.n_atoms

        if verbose > 1:
            dist_mode_str = dist_modes[self.distanceMode]
            msg_str = f"Using distance calculation mode: {dist_mode_str}"
            print(msg_str, flush=True)
            size = int(nAtoms * (nAtoms - 1) / 2) * np.array(0, dtype=float).itemsize
            mbStr = "Allocating temporary distance array of approximate size {} MB."
            print(mbStr.format(size // 1024 // 1024), flush=True)
            print("Starting allocation now!", flush=True)

        # Array to receive all-to-all distances, at every step
        if self.distanceMode == ct.MODE_ALL:
            # This array will store all distances between all atoms.
            tmpDists: npt.NDArray = np.zeros(int(nAtoms * (nAtoms - 1) / 2),
                                             dtype=float)
            if verbose > 1:
                print("Filling array with zeros.")

        elif self.distanceMode == ct.MODE_CAPPED:
            # This array will only be modified to store distances shorter
            # than the cutoff. Since all other distances are larger, we
            # initialize the array with an arbitrarily large value.
            fill_val = self.cutoffDist * 2
            tmpDists = np.full(int(nAtoms * (nAtoms - 1) / 2),
                               fill_val,
                               dtype=float)
            if verbose > 1:
                print(f"Filling array twice the cutoff distance {fill_val}.")

        if verbose > 1:
            alcStr = "Allocated temporary distance array of size {} MB."
            print(alcStr.format(tmpDists.nbytes // 1024 // 1024), flush=True)

        # Array to get minimum distances per node
        tmpDistsAtms: npt.NDArray = np.full(nAtoms, self.cutoffDist * 2, dtype=float)

        if verbose > 1:
            alcStr = "Allocated temporary NODE distance array of size {} KB."
            print(alcStr.format(tmpDistsAtms.nbytes // 1024), flush=True)

        if 0 > end:
            end = self.workU.trajectory.n_frames

        if verbose > 1:
            print(f"Checking frames {beg} to {end} with stride {stride}.")

        for ts in self.workU.trajectory[beg:end:stride]:

            if verbose > 1:
                print("Calculating contacts for timestep {}.".format(ts),
                      flush=True)

            # Calculates distances to determine contact matrix
            ct.get_contacts_c(selectionAtms,
                              self.numNodes,
                              nAtoms,
                              self.cutoffDist,
                              tmpDists,
                              tmpDistsAtms,
                              contact_mat,
                              self.atomToNode,
                              self.nodeGroupIndicesNP,
                              self.nodeGroupIndicesNPAux,
                              dist_mode=self.distanceMode)

    def findContacts(self, stride: int = 1, verbose: int = 1) -> None:
        """Finds all nodes in contact.

        This is the main user interface access to calculate nodes in contact.
        This function automatically splits the whole trajectory into windows,
        allocates NumPy array objects to speed up calculations, and leverages
        MDAnalysis parallel implementation to determine atom distances.

        After determining which frames of a trajectory window show contacts
        between atom groups, it checks the contact cutoff to determine if two
        nodes has enough contact during a simulation window to be considered
        "in contact". A final contact matrix is created and stored in the
        DNAproc object.

        This function automatically updates the unified contact matrix that
        displays nodes in contact in *any* simulation window.
        The function also performs general sanity checks by
        calling :py:func:`checkContactMat`.

        Args:
            stride (int) : Controls how many trajectory frames will be skipped
                during contact calculation.

            verbose (int): Controls verbosity level in the function.

        """

        assert isinstance(stride, int)
        assert isinstance(verbose, int)

        assert stride > 0

        # Allocate contact matrix(ces)
        self.contactMatAll = np.zeros([self.numWinds, self.numNodes, self.numNodes],
                                      dtype=np.int64)

        # Set number of frames per window.
        winLen = int(np.floor(self.workU.trajectory.n_frames / self.numWinds))

        # Set number of frames that defines a contact
        contactCutoff = (winLen / stride) * self.contactPersistence

        for winIndx in self.progBar(range(self.numWinds),
                                    total=self.numWinds,
                                    desc="Window",
                                    ascii=self.asciiMode):
            beg = winIndx * winLen
            end = (winIndx + 1) * winLen

            if verbose > 1:
                msgStr = f"Starting contact calculation for window {winIndx} ..."
                print(msgStr, flush=True)
                start_timer = timer()

            self._contactTraj(self.contactMatAll[winIndx, :, :],
                              beg,
                              end,
                              stride,
                              verbose)

            if verbose > 1:
                end_timer = timer()
                time_delta = end_timer - start_timer
                timeStr = "Time for contact calculation: {}"
                print(timeStr.format(timedelta(seconds=time_delta)), flush=True)

            self.contactMatAll[winIndx, :, :] = \
                np.where(self.contactMatAll[winIndx, :, :] > contactCutoff, 1, 0)

            # Contact is calculated in Node_i to Node_j format, where i < j. This
            # the following loop populates the other half of the matrix for j < i.
            for i in range(self.numNodes):
                for j in range(i + 1, self.numNodes):
                    self.contactMatAll[winIndx, j, i] = \
                        self.contactMatAll[winIndx, i, j]

        # Update unified contact matrix for all windows
        self._genContactMatrix()

        self.checkContactMat(verbose=verbose)

    def _genContactMatrix(self) -> None:
        """Update unified contact matrix for all windows.
        """
        # Allocate a unified contact matrix for all windows.
        self.contactMat = np.zeros([self.numNodes, self.numNodes], dtype=np.int64)

        # Join all data for all windows on the same unified contact matrix.
        for winIndx in range(self.numWinds):
            self.contactMat = np.add(self.contactMat, self.contactMatAll[winIndx, :, :])

        # Creates binary mask with pairs of nodes in contact
        self.contactMat = np.where(self.contactMat > 0, 1, 0)

    def checkContactMat(self, verbose: int = 1) -> None:
        """Sanity checks for contact matrix for all windows.

        Checks if the contact matrix is symmetric and if there are any nodes
        that make no contacts to any other nodes across all windows.
        The function also calculates the percentage of nodes in contact over
        the entire system.

        Args:
            verbose (bool) : Controls how much output will the function print.

        """

        assert isinstance(verbose, int)

        # Sanity Checks that the contact matrix is symmetric
        if not np.allclose(self.contactMat, self.contactMat.T, atol=0.1):
            raise Exception("ERROR: the contact matrix is not symmetric.")

        # Checks if there is any node that does not make contacts to ANY other node.
        noContactNodes = np.asarray(np.where(np.sum(self.contactMat, axis=1) == 0)[0])
        if verbose:
            print("We found {0} nodes with no contacts.".format(len(noContactNodes)))

        # Counts total number of contacts
        if verbose:
            pairs = np.asarray(np.where(np.triu(self.contactMat) > 0)).T
            totalPairs = self.contactMat.shape[0] * (self.contactMat.shape[0] - 1)
            totalPairs = int(totalPairs / 2)
            verbStr = "We found {0:n} contacting pairs out of {1:n} " \
                      "total pairs of nodes."
            print(verbStr.format(len(pairs), totalPairs))
            pairPc = round((len(pairs) / totalPairs) * 100, 1)
            print("(That's {0}%, by the way)".format(pairPc))

    # TODO: reduce complexity - Flake8 marks it at 22
    def _remove_isolated(self, verbose: int = 0) -> None:  # noqa: C901

        if verbose > 0:
            print("\nRemoving isolated nodes...\n")

        # Update unified contact matrix for all windows
        self._genContactMatrix()

        # Gets indices for nodes with no contacts
        noContactNodesArray = np.sum(self.contactMat, axis=1) == 0
        contactNodesArray = ~noContactNodesArray

        # Atom selection for nodes with contact
        contactNodesSel = self.nodesAtmSel.atoms[contactNodesArray]
        noContactNodesSel = self.nodesAtmSel.atoms[noContactNodesArray]

        # Keep nodes that belong to residues with at least one network-bound node.
        # This is important in lipids and nucleotides that may have multiple nodes,
        # and only one is connected to the system.
        # First we select *residues* which have at least one node in contact.
        resNoContacts = list(set(noContactNodesSel.residues.ix) -
                             set(contactNodesSel.residues.ix))
        # Then we create an atom selection for residues with no nodes in contact.
        noContactNodesSel = self.nodesAtmSel.residues[resNoContacts]
        # Finally we update the selection for all nodes that belong to residues with
        # at least one node in contact.
        mask: npt.NDArray = np.ones(len(self.nodesAtmSel.residues), dtype=bool)
        mask[resNoContacts] = False
        contactNodesSel = self.nodesAtmSel.residues[mask].atoms.intersection(
            self.nodesAtmSel)

        # We also have to update the contact matrix that represent nodes which will
        # be kept in the system. For this we will build another mask.
        nodeMask: npt.NDArray = np.ones(len(self.nodesAtmSel.atoms.ids), dtype=bool)
        for indx, atm in enumerate(self.nodesAtmSel.atoms):
            # We check if the atom belongs to the selection of atoms that will be
            # kept in the system.
            nodeMask[indx] = atm.id in contactNodesSel.atoms.ids

        # Checks if there is any node that does not make contacts to ANY other node.
        if verbose > 0:
            print("We found {0} nodes with no contacts.".format(len(noContactNodesSel)))
        if verbose > 1:
            for atm in noContactNodesSel.atoms:
                print(atm)

        # Trims matrices
        self.contactMatAll = self.contactMatAll[:, nodeMask, :]
        self.contactMatAll = self.contactMatAll[:, :, nodeMask]

        self.contactMat = self.contactMat[nodeMask, :]
        self.contactMat = self.contactMat[:, nodeMask]

        if verbose > 0:
            if len(noContactNodesSel):
                statusStr = "\nIsolated nodes removed. We now have {} nodes in " \
                            "the system\n"
                print(statusStr.format(self.contactMatAll[0].shape[0]))
            print("Running new contact matrix sanity check...")

        self.checkContactMat(verbose)

        #########################
        # Update Universe and network data

        if verbose > 0 and len(noContactNodesSel):
            print("\nUpdating Universe to reflect new node selection...")

        # Here we use the new node selection to find *all atoms* from residues
        # that contain selected nodes, not just the atoms that represent
        # nodes stored in the `contactNodesSel` variable.

        # Instead of using a long selection string programmatically created
        # for all nodes, the following loop gathers all selected residues
        # from all segments that have at least one selected residue. Then,
        # it creates a list with one string per segment. This reduces the
        # load on the recursion-based atom selection language in MDanalysis.

        from collections import defaultdict
        selDict = defaultdict(list)
        for res in contactNodesSel.residues:
            selDict[res.segid].append(str(res.resid))

        selStrL = []
        for segid, residL in selDict.items():
            selStrL.append("segid {} and resid {}".format(segid, " ".join(residL)))

        if verbose > 1:
            print("Creating a smaller atom selection without isolated nodes.")

        allSel = self.workU.select_atoms(*selStrL)

        if verbose > 1:
            print("Creating a smaller universe without isolated nodes.")

        # Merging a selection from the universe returns a new (and smaller) universe
        self.workU = mda.core.universe.Merge(allSel)

        if verbose > 1:
            print("Capture coordinates from selected nodes in previous universe.")

        # We now create a new universe with coordinates from the selected residues
        resObj = mdaAFF(lambda ag: ag.positions.copy(), allSel).run().results

        # This checks the type of the MDAnalysis results. Prior to version 2.0.0,
        # MDA returned a numpy.ndarray with the trajectory coordinates. After
        # version 2.0.0, it returns a results object that contains the trajectory.
        # With this check, the code can handle both APIs.
        if not isinstance(resObj, np.ndarray):
            resObj = resObj['timeseries']

        if verbose > 1:
            print("Load coordinates from selected nodes in new universe.")

        self.workU.load_new(resObj, format=mdaMemRead)

        if verbose > 1:
            print("Recreate node-atom selection.")

        # Regenerate selection of atoms that represent nodes.

        # Builds list of selection statements
        selStrL = ["(resname {0} and name {1})".format(k, " ".join(v.keys()))
                   for k, v in self.resNodeGroups.items()]

        # Combines all statements into one selection string
        selStr = " or ".join(selStrL)

        if verbose > 1:
            print("Selection string for atoms that represent network nodes:")
            print(selStr)

        self.nodesAtmSel = self.workU.select_atoms(selStr)

        self.numNodes = self.nodesAtmSel.n_atoms

        # Creates an array relating all atoms in the system to nodes.
        self.atomToNode = np.full(shape=allSel.n_atoms,
                                  fill_value=-1,
                                  dtype=int)

        if verbose > 0:
            print("Updating atom-to-node mapping...")

        for indx, node in enumerate(self.progBar(self.nodesAtmSel.atoms,
                                                 desc="Node",
                                                 ascii=self.asciiMode)):

            # Loops over all atoms in the residue related to the node
            for atm in node.residue.atoms:

                # Checks if the atom name is listed for the node
                if atm.name in self.resNodeGroups[node.resname][node.name]:
                    self.atomToNode[atm.ix] = indx

        self._check_atom_to_node_mapping(verbose)

    def filterContacts(self,
                       notSameRes: bool = True,
                       notConsecutiveRes: bool = False,
                       removeIsolatedNodes: bool = True,
                       verbose: int = 1) -> None:
        """Filters network contacts over the system.

        The function removes edges and nodes in preparation for network analysis.
        Traditionally, edges between nodes within the same residue are removed,
        as well as edges between nodes in consecutive residues within the same
        polymer chain (protein or nucleic acid). Essentially, nodes that have
        covalent bonds connecting their node groups can bias the analysis and
        hide important non-bonded interactions.

        The function also removes nodes that are isolated and make no contacts
        with any other nodes. Examples are ions or solvent residues that were
        initially included in the system through the preliminary automated solvent
        detection routine, but did not reach the contact threshold for being
        part of the final system.

        After filtering nodes and edges, the function updates the MDAnalysis
        universe and network data.

        Args:
            notSameRes (bool) : Remove contacts between nodes in the same residue.
            notConsecutiveRes (bool) : Remove contacts between nodes in
                consecutive residues.
            removeIsolatedNodes (bool) : Remove nodes with no contacts.
            verbose (bool) : Controls verbosity of output.
        """

        assert isinstance(notSameRes, bool)
        assert isinstance(notConsecutiveRes, bool)
        assert isinstance(removeIsolatedNodes, bool)
        assert isinstance(verbose, int)

        if verbose > 0:
            print("Filtering contacts in each window.")

        for winIndx in self.progBar(range(self.numWinds), total=self.numWinds,
                                    desc="Window", ascii=self.asciiMode):
            self._filterContactsWindow(self.contactMatAll[winIndx, :, :],
                                       notSameRes=notSameRes,
                                       notConsecutiveRes=notConsecutiveRes,
                                       verbose=verbose)

            if verbose > 1:
                print("Window:", winIndx)

            upTri = np.triu(self.contactMatAll[winIndx, :, :])
            pairs = np.asarray(np.where(upTri > 0)).T
            totalPairs = self.contactMat.shape[0] * (self.contactMat.shape[0] - 1)
            totalPairs = int(totalPairs / 2)

            if verbose > 1:
                verbStr = "We found {0:n} contacting pairs out of {1:n} " \
                          "total pairs of nodes."
                print(verbStr.format(len(pairs), totalPairs))
                pairPc = round((len(pairs) / totalPairs) * 100, 1)
                print("(That's {0}%, by the way)".format(pairPc))

        if removeIsolatedNodes:
            self._remove_isolated(verbose)

    def _filterContactsWindow(self,
                              mat: np.ndarray,
                              notSameRes: bool = True,
                              notConsecutiveRes: bool = False,
                              verbose: int = 0) -> None:
        """Filter contacts in a contact matrix.

        This function receives a contact matrix and guarantees that there will
        be no self-contacts (a results of some contact detection algorithms).
        Optionally, it can also erase contacts between nodes in the same
        residue (notSameRes) and between nodes in consecutive
        residues (notConsecutiveRes).

        Args:
            notSameRes (bool) : Remove contacts between nodes in the same residue.
            notConsecutiveRes (bool) : Remove contacts between nodes in
                consecutive residues.

        """

        # Cycles over all nodes in the system. There may be several nodes per residue.
        for node in self.progBar(self.nodesAtmSel.atoms,
                                 desc="Node",
                                 leave=(verbose > 1),
                                 ascii=self.asciiMode):

            # Get current node index
            nodeIndx = self.atomToNode[node.ix]
            # Get current node residue
            res = node.residue

            # No contact between the same node (main diagonal)
            mat[nodeIndx, nodeIndx] = 0

            # No contact between nodes in same residue
            if notSameRes:
                # Get all node atom(s) in current residue
                resSelection = self.workU.select_atoms(
                    "(segid {0} and resid {1})".format(res.segid, res.resid) +
                    " and name " +
                    " ".join(self.resNodeGroups[res.resname].keys())
                )
                # Get their node indices
                nodeIndxs = self.atomToNode[resSelection.atoms.ix_array]
                # No correlation between nodes in the same residue.
                # Also zeroes self-correlation.
                for i, j in [(i, j) for i in nodeIndxs for j in nodeIndxs]:
                    mat[i, j] = 0

            # Keeps node from making direct contact to previous residue in chain
            if notConsecutiveRes and (res.resindex - 1 >= 0):
                prevRes = self.workU.residues[res.resindex - 1]

                if prevRes.segid == res.segid:

                    # Select previous residue in the chain
                    prevResSel = self.workU.select_atoms(
                        "(segid {0} and resid {1})".format(res.segid, prevRes.resid)
                    )
                    # Select the node atom(s) from previous residue
                    prevSelStr = "name " + \
                                 " ".join(self.resNodeGroups[prevRes.resname].keys())
                    prevResSel = prevResSel.select_atoms(prevSelStr)

                    # Checks that it is not an ION or Water residue from same segment
                    if prevRes.atoms.n_atoms > 1:
                        # Get the actual node(s) indice(s) from the previous residue
                        nodeIndxs = self.atomToNode[prevResSel.atoms.ix_array]

                        # Zeroes all correlation between nodes in consecutive residues
                        for trgtIndx in nodeIndxs:
                            mat[nodeIndx, trgtIndx] = 0
                            mat[trgtIndx, nodeIndx] = 0

            # Keeps node from making direct contact to following residue in chain
            # (same as above)
            if notConsecutiveRes and (res.resindex + 1 < self.workU.residues.n_residues):
                folRes = self.workU.residues[res.resindex + 1]

                if folRes.segid == res.segid:

                    folResSel = self.workU.select_atoms(
                        "(segid {0} and resid {1})".format(res.segid, folRes.resid)
                    )
                    folResStr = "name " + \
                                " ".join(self.resNodeGroups[folRes.resname].keys())
                    folResSel = folResSel.select_atoms(folResStr)

                    if folRes.atoms.n_atoms > 1:
                        nodeIndxs = self.atomToNode[folResSel.atoms.ix_array]

                        for trgtIndx in nodeIndxs:
                            mat[nodeIndx, trgtIndx] = 0
                            mat[trgtIndx, nodeIndx] = 0

    def _prep_phi_psi(self, win_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Pre-calculate Phi and Psi

        Args:
            win_len: number of frames in each trajectory window

        Returns:
            Pre-calculated psi and spi arrays.
        """

        # Pre-calculate psi values for all frames.
        # (allocation and initialization step)
        psi: npt.NDArray = np.zeros([win_len + 1], dtype=np.float64)
        psi[1] = -0.57721566490153

        # Pre-calculate psi values for all frames.
        # (actual calculation step)
        for indx in range(win_len):
            if indx > 0:
                psi[indx + 1] = psi[indx] + 1 / indx

        # Pre calculates "psi[k] - 1/k"
        phi: np.ndarray = np.ndarray([self.kNeighb + 1], dtype=np.float64)
        for indx in range(1, (self.kNeighb + 1)):
            phi[indx] = psi[indx] - 1 / indx

        return psi, phi

    def _create_pair_list(self, win_indx: int, verbose: int = 0) -> np.ndarray:
        """ Creates list of pairs of nodes in contact.

        The list is ordered to reduce the frequency with which parallel
        calculations will read the same trajectory (access the same memory).
        The method will also remove pairs for which we already have correlations.

        Args:
            win_indx (int) : Defines the window used to find contacts.
            verbose (int) : Defines verbosity of output.

        Returns:

        """

        assert isinstance(win_indx, int)
        assert isinstance(verbose, int)

        pairList: list = []

        # Build pair list avoiding overlapping nodes (which would require
        # reading the same trajectory).
        for diag in range(1, self.numNodes):
            contI = 0
            contJ = diag
            while contJ < self.numNodes:
                if self.contactMatAll[win_indx, contI, contJ]:
                    pairList.append([contI, contJ])
                contI += 1
                contJ += 1

        # Removes pairs of nodes that already have a result
        # ATTENTION: Contacts that had zero correlation will be recalculated!
        upTri = np.triu(self.corrMatAll[win_indx, :, :])
        upTriT = np.asarray(np.where(upTri > 0)).T
        precalcPairList = upTriT.tolist()

        if 0 < len(precalcPairList):

            if verbose > 0:
                verStr = f"Removing {len(precalcPairList)} pairs with " \
                         f"pre-calculated correlations in window {win_indx}."
                print(verStr)

            pairList = [pair for pair in pairList
                        if pair not in precalcPairList]

        return np.asarray(pairList)

    # TODO: reduce complexity - Flake8 marks it at 16
    def calcCor(self,  # noqa: C901
                ncores: int = 1,
                forceCalc: bool = False,
                verbose: int = 0) -> None:
        """Main interface for correlation calculation.

        Calculates generalized correlation coefficients either in serial
        or in parallel implementations using Python's multiprocessing
        package. This function wraps the creation of temporary variables
        in allocates the necessary NumPy arrays for accelerated
        performance of MDAnalysis algorithms.

        .. note:: See also :py:func:`~dynetan.gencor.prep_mi_c`.
        .. note:: See also :py:func:`~dynetan.gencor.calc_mir_numba_2var`.
        .. note:: See also :py:func:`~dynetan.gencor.calc_cor_proc`.

        Args:
            ncores (int) : Defines how many cores will be used for
                calculation of generalized correlation coefficients. Set to
                `1` in order to use the serial implementation.

            forceCalc (bool) : Defines if correlations will be calculated again
                even if they have been calculated before.
        """

        assert isinstance(ncores, int)
        assert isinstance(forceCalc, bool)
        assert isinstance(verbose, int)

        assert ncores >= 1, "Number of cores must be at least 1."

        # For 3D atom position data
        num_dims = 3

        if verbose > 0:
            print("Calculating correlations...\n")

        win_len = int(np.floor(self.workU.trajectory.n_frames / self.numWinds))

        if verbose > 0:
            print(f"Using window length of {win_len} simulation steps.")

        # Check if the correlation matrix already exists (if this function is
        #   being executed again on the same system, with updated contacts), or
        #   if all correlations should be recalculated from scratch (in case the
        #   trajectory or contact matrix was modified in some way).
        if (self.corrMatAll is None) or forceCalc:
            # Initialize the correlation matrix with zeros
            self.corrMatAll = np.zeros([self.numWinds, self.numNodes, self.numNodes],
                                       dtype=np.float64)

        # Stores all data in a dimension-by-frame format.
        traj: np.ndarray = np.ndarray([self.numNodes, num_dims, win_len],
                                      dtype=np.float64)

        psi, phi = self._prep_phi_psi(win_len)

        for winIndx in self.progBar(range(self.numWinds),
                                    total=self.numWinds,
                                    desc="Window",
                                    ascii=self.asciiMode):
            beg = int(winIndx * win_len)
            end = int((winIndx + 1) * win_len)

            pair_array = self._create_pair_list(winIndx, verbose)

            if pair_array.shape[0] == 0:
                if verbose > 0:
                    print(f"No new correlations to be calculated "
                          f"in window {winIndx}.")
                break
            else:
                if verbose > 0:
                    print(f"{pair_array.shape[0]} new correlations to "
                          f"be calculated in window {winIndx}.")

            # Resets the trajectory NP array for the current window.
            traj.fill(0)

            # Prepares data for fast calculation of the current window.
            gc.prep_mi_c(self.workU, traj, beg, end, self.numNodes, num_dims)

            if ncores == 1:

                if verbose > 0:
                    print("- > Using single-core implementation.")

                # Iterates over all pairs of nodes that are in contact.
                for atmList in self.progBar(pair_array,
                                            desc="Contact Pair",
                                            leave=False,
                                            ascii=self.asciiMode):

                    # Calls the Numba-compiled function.
                    mir = gc.calc_mir_numba_2var(traj[atmList, :, :],
                                                 win_len,
                                                 num_dims,
                                                 self.kNeighb,
                                                 psi,
                                                 phi)

                    corr = gc.mir_to_corr(mir)

                    self.corrMatAll[winIndx, atmList[0], atmList[1]] = corr
                    self.corrMatAll[winIndx, atmList[1], atmList[0]] = corr

            else:

                if verbose > 0:
                    print(f"- > Using multi-core implementation with {ncores} threads.")

                # Create queues that feed processes with node pairs, and gather results.
                data_queue: queue.Queue = mp.Queue()
                results_queue: queue.Queue = mp.Queue()

                # Loads the node pairs in the input queue
                for atmList in pair_array:
                    data_queue.put(atmList)

                # Creates processes.
                procs = []
                for _ in range(ncores):

                    # Include termination flags for the processes in the input queue
                    # The termination flag is an empty list
                    data_queue.put([])

                    # Initialize process
                    proc = mp.Process(target=gc.calc_cor_proc,
                                      args=(traj,
                                            win_len,
                                            psi,
                                            phi,
                                            num_dims,
                                            self.kNeighb,
                                            data_queue, results_queue)
                                      )
                    proc.start()
                    procs.append(proc)

                # Gathers all results.
                for _ in self.progBar(range(len(pair_array)),
                                      desc="Contact Pair",
                                      leave=False,
                                      ascii=self.asciiMode):
                    # Waits until the next result is available,
                    # then puts it in the matrix.
                    result = results_queue.get()

                    node1 = result[0][0]
                    node2 = result[0][1]
                    corr = gc.mir_to_corr(result[1])

                    self.corrMatAll[winIndx, node1, node2] = corr
                    self.corrMatAll[winIndx, node2, node1] = corr

                # Joins processes.
                for proc in procs:
                    proc.join()

        self._corr_mat_symmetric()

    def _corr_mat_symmetric(self) -> None:
        # Sanity Check: Checks that the correlation and contact matrix is symmetric
        for win in range(self.numWinds):
            if not np.allclose(self.corrMatAll[win, :, :],
                               self.corrMatAll[win, :, :].T,
                               atol=0.1):
                err_str = f"ERROR: Correlation matrix for window {win} is " \
                          f"NOT symmetric!!"
                raise Exception(err_str)

    def calcCartesian(self,
                      backend: backend_types_literal = "serial",
                      verbose: int = 1) -> None:
        """Main interface for calculation of cartesian distances.

        Determines the shortest cartesian distance between atoms in node groups
        of all network nodes. Using a sampling of simulation frames, the function
        also calculates statistics on such measures, including mean distance,
        standard error of the mean, minimum, and maximum.
        This allows analysis comparing network distances and cartesian distances.

        .. note:: See also :py:func:`~dynetan.contact.calc_distances`
            and :py:func:`~dynetan.toolkit.getCartDist`.

        Args:
            backend (str) : Defines which MDAnalysis backend will be used for
                calculation of cartesian distances. Options are `serial` or
                `openmp`. This option is ignored if the distance mode is not "all".
            verbose (int) : Defines verbosity of output.
        """

        assert isinstance(verbose, int)
        assert isinstance(backend, str)

        assert backend in backend_types, f"Only allowed backend options " \
                                         f"are {backend_types}"

        if verbose > 0:
            print("Calculating cartesian distances...\n")

        # numFramesDists is used in the calculation of statistics!
        numFramesDists = self.numSampledFrames * self.numWinds

        selectionAtms = self.workU.select_atoms("all")

        array_size = int(self.numNodes * (self.numNodes - 1) / 2)
        nodeDistsTmp: npt.NDArray = np.zeros([array_size], dtype=np.float64)

        self.nodeDists = np.zeros([4, int(self.numNodes * (self.numNodes - 1) / 2)],
                                  dtype=np.float64)

        if verbose > 0:
            outStr = "Sampling a total of {0} frames from " \
                     "{1} windows ({2} per window)..."
            print(outStr.format(numFramesDists, self.numWinds, self.numSampledFrames))

        steps = int(np.floor(len(self.workU.trajectory) / numFramesDists))
        maxFrame = numFramesDists * steps

        # Mean distance
        for _ in self.progBar(self.workU.trajectory[0:maxFrame:steps],
                              total=numFramesDists,
                              desc="MEAN: Timesteps",
                              ascii=self.asciiMode):
            ct.calc_distances(selectionAtms,
                              self.numNodes,
                              selectionAtms.n_atoms,
                              self.atomToNode,
                              self.cutoffDist,
                              self.nodeGroupIndicesNP,
                              self.nodeGroupIndicesNPAux,
                              nodeDistsTmp,
                              backend,
                              dist_mode=self.distanceMode,
                              verbose=verbose)

            # Mean
            self.nodeDists[0, :] += nodeDistsTmp

        self.nodeDists[0, :] /= numFramesDists

        # Initializes the min and max distances with the means.
        self.nodeDists[2, :] = self.nodeDists[0, :]
        self.nodeDists[3, :] = self.nodeDists[0, :]

        # Standard Error of the Mean
        for _ in self.progBar(self.workU.trajectory[0:maxFrame:steps],
                              total=numFramesDists,
                              desc="SEM/MIN/MAX: Timesteps",
                              ascii=self.asciiMode):
            ct.calc_distances(selectionAtms,
                              self.numNodes,
                              selectionAtms.n_atoms,
                              self.atomToNode,
                              self.cutoffDist,
                              self.nodeGroupIndicesNP,
                              self.nodeGroupIndicesNPAux,
                              nodeDistsTmp,
                              backend,
                              dist_mode=self.distanceMode,
                              verbose=verbose)

            # Accumulates the squared difference
            self.nodeDists[1, :] += np.square(self.nodeDists[0, :] - nodeDistsTmp)

            # Checks for the minimum
            self.nodeDists[2, :] = np.where(nodeDistsTmp < self.nodeDists[2, :],
                                            nodeDistsTmp, self.nodeDists[2, :])

            # Checks for the maximum
            self.nodeDists[3, :] = np.where(nodeDistsTmp > self.nodeDists[3, :],
                                            nodeDistsTmp, self.nodeDists[3, :])

        if numFramesDists > 1:
            # Sample standard deviation: SQRT of sum divided by N-1
            self.nodeDists[1, :] = np.sqrt(self.nodeDists[1, :] / (numFramesDists - 1))
            # SEM:  STD / sqrt(N)
            self.nodeDists[1, :] = self.nodeDists[1, :] / np.sqrt(numFramesDists)

    def calcGraphInfo(self) -> None:
        """Create a graph from the correlation matrix.

        Uses NetworkX to create a graph representation of the network.
        One graph is created per simulation window.

        For network analysis, node `distances` are generated with a log
        transformation of the correlation values. This way, edges between nodes
        with higher correlation coefficients are considered "closer", with
        shorter distances, and nodes with low correlation coefficients are
        "far apart", with larger distance.

        .. note:: See also :py:func:`~dynetan.network.calcOptPathPar`
            and :py:func:`~dynetan.network.calcBetweenPar`.

        """
        self.nxGraphs = []

        for win in range(self.numWinds):
            self.nxGraphs.append(nx.Graph(self.corrMatAll[win, :, :]))

            # We substitute zeros for a non-zero value to avoid "zero division" warnings
            #   from the np.log transformation below.
            self.corrMatAll[np.where(self.corrMatAll == 0)] = 10 ** -11

            # Use log transformation for network distance calculations.
            tmpLogTransf = -1.0 * np.log(self.corrMatAll[win, :, :])

            # Now we guarantee that the previous transformation does not
            #   create "negative infitite" distances. We set those to zero.
            tmpLogTransf[np.where(np.isinf(tmpLogTransf))] = 0

            # Now we return to zero-correlation we had before.
            self.corrMatAll[np.where(self.corrMatAll < 10 ** -10)] = 0

            # Loop over all graph edges and set their distances.
            for pair in self.nxGraphs[win].edges.keys():
                self.nxGraphs[win].edges[(pair[0], pair[1])]["dist"] = \
                    tmpLogTransf[pair[0], pair[1]]

            # Sets the degree of each node.
            degree_dict = dict(self.nxGraphs[win].degree(self.nxGraphs[win].nodes()))
            nx.set_node_attributes(self.nxGraphs[win], degree_dict, 'degree')

    def getDegreeDict(self, window: int = 0) -> dict:
        """Compiles a dictionary with node degrees.

        This wrapper function uses NetworkX graph object to list the
        degrees of all nodes.

        Args:
            window (int) : Simulation window.

        """
        assert isinstance(window, int)

        if window < 0 or window >= self.numWinds:
            raise Exception(f"Requested window {window} out of range.")

        return dict(self.nxGraphs[window].degree(self.nxGraphs[window].nodes()))

    def calcOptPaths(self, ncores: int = 1) -> None:
        """Main interface for optimal path calculations.

        Calculates optimal paths between all nodes in the network using
        NetworkX implementation of the Floyd Warshall algorithm. When using more
        than one core, this function uses Python's `multiprocessing`
        infrastructure to calculate optimal paths in multiple simulation
        windows simultaneously.

        .. note:: See also :py:func:`~dynetan.network.calcOptPathPar`.

        Args:
            ncores (int) : Defines how many cores will be used for calculation
                of optimal paths. Set to `1` in order to use the
                serial implementation.

        """

        assert isinstance(ncores, int)

        if ncores <= 0:
            raise Exception("ERROR: number of cores must be at least 1.")

        # Sets the network distance array.
        self.distsAll = np.zeros([self.numWinds, self.numNodes, self.numNodes],
                                 dtype=np.float64)

        self.preds = {}
        for i in range(self.numWinds):
            self.preds[i] = 0

        if ncores == 1:
            # Serial Version

            for win in self.progBar(range(self.numWinds), total=self.numWinds,
                                    desc="Window", ascii=self.asciiMode):
                # IMPORTANT! #####
                # For the FW optimal path determination, we use the "distance"
                # as weight, that is, the log-transformation of the correlations.
                # NOT the correlation itself.
                pathsPred, pathsDist = nxFWPD(self.nxGraphs[win], weight='dist')

                # Turns dictionary of distances into NumPy 2D array per window
                # Notice the nested list comprehensions.
                self.distsAll[win, :, :] = np.array([
                    [pathsDist[i][j] for i in sorted(pathsDist[j])]
                    for j in sorted(pathsDist)])

                # Combines predecessor dictionaries from all windows
                self.preds[win] = copy.deepcopy(pathsPred)

        else:

            inQueue: queue.Queue = mp.Queue()
            outQueue: queue.Queue = mp.Queue()

            for win in range(self.numWinds):
                inQueue.put(win)

            # Creates processes.
            procs = []
            for _ in range(ncores):

                # Include termination flags for the processes in the input queue
                # The termination flag is an invalid window index of -1.
                inQueue.put(-1)

                procs.append(mp.Process(target=nw.calcOptPathPar,
                                        args=(self.nxGraphs, inQueue, outQueue)))
                procs[-1].start()

            for _ in self.progBar(range(self.numWinds),
                                  total=self.numWinds,
                                  desc="Window",
                                  ascii=self.asciiMode):
                # Waits until the next result is available,
                # then stores it in the object.
                result = outQueue.get()

                self.distsAll[result[0], :, :] = np.copy(result[1])
                self.preds[result[0]] = copy.deepcopy(result[2])

            # Joins processes.
            for proc in procs:
                proc.join()

        # Get maximum network distance
        self.maxDist = np.max(self.distsAll[self.distsAll != np.inf])

        # Set -1 as distance of nodes with no connecting path (instead of np.inf)
        self.distsAll[np.where(np.isinf(self.distsAll))] = -1

        # Maximum network distance between directly connected nodes (path length == 2)
        # We check connection with the correlation matrix because at times,
        # two nodes may be in contact (physical proximity) but may not
        # have any correlation.
        self.maxDirectDist = max([self.distsAll[win, self.corrMatAll[win, :, :] > 0].max()
                                  for win in range(self.numWinds)])

    def getPath(self, node_i: int, node_j: int, window: int = 0) -> list:
        """Wrapper for NetworkX reconstruct_path.

        The function calls NetworkX's *reconstruct_path* to return the list of
        nodes that connect `nodeI` to `nodeJ`. This function must only be called
        **after** a path detection run has been completed
        (see :py:func:`~dynetan.proctraj.DNAproc.calcOptPaths`).

        Args:
            node_i (int) : Node ID.
            node_j (int) : Node ID.
            window (int) : Simulation window.

        Returns:
            List of node IDs.

        """

        assert isinstance(node_i, (int, np.integer))
        assert isinstance(node_j, (int, np.integer))
        assert isinstance(window, (int, np.integer))

        return nx.reconstruct_path(node_i, node_j, self.preds[window])

    def calcBetween(self, ncores: int = 1) -> None:
        """Main interface for betweeness calculations.

        Calculates betweenness for all nodes in the network using NetworkX
        implementation of the betweenness centrality for edges and eigenvector
        centrality for nodes. When using more than one core, this function uses
        Python's `multiprocessing` infrastructure to calculate betweenness in
        multiple simulation windows simultaneously.

        .. note:: See also :py:func:`~dynetan.network.calcBetweenPar`.

        Args:
            ncores (int) : Defines how many cores will be used for calculation.
                Set to `1` in order to use the serial implementation.

        """

        assert isinstance(ncores, int)

        if ncores <= 0:
            raise Exception("ERROR: number of cores must be at least 1.")

        self.btws = {}

        if ncores == 1:
            # Serial Version
            # Single core version
            for win in self.progBar(range(self.numWinds),
                                    total=self.numWinds,
                                    desc="Window",
                                    ascii=self.asciiMode):
                # Calc all betweenness in entire system.
                # IMPORTANT! ##########
                # For the betweenness, we only care about the number of the
                # shortest paths passing through a given edge, so no weights
                # are considered.
                self.btws[win] = nxbetweenness(self.nxGraphs[win], weight=None)

                # Creates an ordered dict of pairs with betweenness higher than zero.
                self.btws[win] = {k: self.btws[win][k] for k in self.btws[win].keys()
                                  if self.btws[win][k] > 0}
                self.btws[win] = OrderedDict(sorted(self.btws[win].items(),
                                                    key=lambda t: t[1],
                                                    reverse=True))
        else:

            inQueue: queue.Queue = mp.Queue()
            outQueue: queue.Queue = mp.Queue()

            for win in range(self.numWinds):
                inQueue.put(win)

            # Creates processes.
            procs = []
            for _ in range(ncores):
                # Include termination flags for the processes in the input queue
                # The termination flag is an invalid window index of -1.
                inQueue.put(-1)

                procs.append(mp.Process(target=nw.calcBetweenPar,
                                        args=(self.nxGraphs, inQueue, outQueue)))
                procs[-1].start()

            for _ in self.progBar(range(self.numWinds),
                                  total=self.numWinds,
                                  desc="Window",
                                  ascii=self.asciiMode):
                # Waits until the next result is available,
                # then stores it in the object.
                result = outQueue.get()

                self.btws[result[0]] = copy.deepcopy(result[1])

            # Joins processes.
            for proc in procs:
                proc.join()

    def calcEigenCentral(self) -> None:
        """Wrapper for calculation of node centrality.

        Calculates node centrality for all nodes in all simulation windows.
        This calculation is relatively inexpensive and is only implemented for
        serial processing.

        All results are stored in the network graph itself.

        """
        for win in range(self.numWinds):
            # Calc all node centrality values in the system.
            cent = nxeigencentrality(self.nxGraphs[win], weight='weight')
            nx.set_node_attributes(self.nxGraphs[win], cent, 'eigenvector')

    def calcCommunities(self) -> None:
        """Calculate node communities using Louvain heuristics.

        The function produces sets of nodes that are strongly connected,
        presenting high correlation coefficients.

        It uses Louvain heuristics as an efficient and precise alternative to
        the classical Girvan–Newman algorithm, which requires much more
        computing power for large and highly connected networks. This method
        also maximizes the modularity of the network. It is inherently random,
        so different calculations performed on the same network data may
        produce slightly different results.

        For more details, see `the original reference
        <http://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta>`_.

        """

        if "eigenvector" in self.nxGraphs[0].nodes[0].keys():
            eigenvAvail = True
        else:
            eigenvAvail = False
            print("WARNING: Node centrality was not calculated (\"eigenvector\" "
                  "attribute not found in graph nodes).")
            print("Nodes will not be ordered by centrality and communities will "
                  "not be ordered by highest node centrality.")

        self.nodesComm = {}

        for win in range(self.numWinds):

            self.nodesComm[win] = {}

            communities = community.best_partition(self.nxGraphs[win])

            comm_arr: npt.NDArray = np.asarray(list(communities.values()), dtype=int)
            communitiesLabels: npt.NDArray = np.unique(comm_arr)

            self.nodesComm[win]["commLabels"] = copy.deepcopy(communitiesLabels)

            nx.set_node_attributes(self.nxGraphs[win], communities, 'modularity')

            self.nodesComm[win]["commNodes"] = {}

            for comm in communitiesLabels:
                # First get a list of just the nodes in that class
                nodesInClass = [n for n in self.nxGraphs[win].nodes()
                                if self.nxGraphs[win].nodes[n]['modularity'] == comm]

                if eigenvAvail:
                    # Then create a dictionary of the eigenvector centralities
                    # of those nodes
                    nodesInClassEigenVs = {n: self.nxGraphs[win].nodes[n]['eigenvector']
                                           for n in nodesInClass}

                    # Then sort that dictionary
                    nodesInClassEigenVsOrd = sorted(nodesInClassEigenVs.items(),
                                                    key=itemgetter(1),
                                                    reverse=True)
                    nodesInClass = [x[0] for x in nodesInClassEigenVsOrd]

                self.nodesComm[win]["commNodes"][comm] = copy.deepcopy(nodesInClass)

            # Orders communities based on size.
            communitiesOrdSize = list(sorted(self.nodesComm[win]["commNodes"].keys(),
                                             key=lambda k: len(
                                                 self.nodesComm[win]["commNodes"][k]),
                                             reverse=True))

            self.nodesComm[win]["commOrderSize"] = copy.deepcopy(communitiesOrdSize)

            def getEgnCentr(comm_id: int) -> Any:
                """
                    Auxiliary function that returns Eigenvector Centralities for
                    nodes of a given community.

                Args:
                    comm_id: Community ID.

                Returns:
                    Centralities of nodes in the community.
                """
                nodes = self.nodesComm[win]["commNodes"][comm_id][0]
                return self.nxGraphs[win].nodes[nodes]['eigenvector']

            if eigenvAvail:
                # Orders communities based on highest eigenvector centrality of
                # all its nodes.
                communitiesOrdEigen = sorted(self.nodesComm[win]["commNodes"].keys(),
                                             key=lambda k: getEgnCentr(k),
                                             reverse=True)
                communitiesOrdEigen = list(communitiesOrdEigen)

                self.nodesComm[win]["commOrderEigenCentr"] = \
                    copy.deepcopy(communitiesOrdEigen)

    def interfaceAnalysis(self,
                          selAstr: str,
                          selBstr: str,
                          betweenDist: float = 15.0,
                          samples: int = 10,
                          verbose: int = 0) -> int:
        """Detects interface between molecules.

        Based on user-defined atom selections, the function detects residues
        (and their network nodes) that are close to the interface between both
        atom selections. That may include amino acids in the interface, as
        well as ligands, waters and ions.

        Only nodes that have edges to nodes on the side of the interface
        are selected.

        Using a sampling of simulation frames assures that transient contacts
        will be detected by this analysis.

        Args:
            selAstr (str) : Atom selection.
            selBstr (str) : Atom selection.
            betweenDist (float) : Cutoff distance for selection of atoms that are
                within *betweenDist* from both selections.
            samples (int) : Number of frames to be sampled for detection of
                interface residues.
            verbose (int) : Controls verbosity of output.

        Returns:
            Number of unique nodes in interface node pairs.

        """

        assert isinstance(selAstr, str)
        assert isinstance(selBstr, str)
        assert isinstance(betweenDist, (float, int))
        assert isinstance(samples, int)
        assert isinstance(verbose, int)

        assert betweenDist > 0
        assert samples > 0

        # Select the necessary stride so that we get *samples*.
        stride = int(np.floor(len(self.workU.trajectory) / samples))

        selPtn = self.workU.select_atoms(selAstr)
        selNcl = self.workU.select_atoms(selBstr)

        contactNodes = set()

        # Find selection of atoms that are within "betweenDist" from both selections.
        # Get selection of nodes represented by the atoms by sampling several frames.
        for ts in self.progBar(self.workU.trajectory[:samples * stride:stride],
                               desc="Samples",
                               total=samples,
                               ascii=self.asciiMode):

            contactSel = mdaB(self.workU.select_atoms("all"),
                              selPtn,
                              selNcl,
                              betweenDist)

            if not isinstance(contactSel, mda.AtomGroup):
                # For MDAnalysis versions older than 2.4:
                # This checks the type of the MDAnalysis results. If the selection
                # or between distance lead to a NULL selection, the function returns
                # 0 ("zero"), otherwise, it returns an "AtomGroup" instance.
                if verbose > 1:
                    warn_str = f"No contacts found in timestep {ts.time}"
                    print(warn_str)
            elif isinstance(contactSel, mda.AtomGroup) and (contactSel.n_atoms == 0):
                # For MDAnalysis versions 2.4 and newer:
                # The between method always returns an AtomGroup
                if verbose > 1:
                    warn_str = f"No contacts found in timestep {ts.time}"
                    print(warn_str)
            else:
                contactNodes.update(np.unique(
                    self.atomToNode[contactSel.atoms.ix_array]))

        if len(contactNodes) == 0:
            if verbose > 0:
                print("No contacts found in this interface. "
                      "Check your selections and sampling.")
            return 0

        # Makes it into a list for better processing
        contactNodesL = np.asarray(list(contactNodes))

        # These are all nodes in both selections.
        numContactNodesL = len(contactNodes)

        # Filter pairs of nodes that have contacts
        pairs_list = []
        for i in range(numContactNodesL):
            for j in range(i, numContactNodesL):
                nodeI = contactNodesL[i]
                nodeJ = contactNodesL[j]
                if max([self.corrMatAll[win, nodeI, nodeJ]
                        for win in range(self.numWinds)]) > 0:
                    pairs_list.append((nodeI, nodeJ))

        # These are all pairs of nodes that make direct connections.
        # These pairs WILL INCLUDE pairs where both nodes are on the same
        # side of the interface.
        contactNodePairs: npt.NDArray = np.asarray(pairs_list, dtype=np.int64)

        # These are pairs where the nodes are NOT on the same selection,
        # that is, pairs that connect the two atom selections.
        self.interNodePairs = [(i, j) for i, j in contactNodePairs
                               if in_interface(self.nodesAtmSel, i, j, self.segIDs)]
        self.interNodePairs = np.asarray(self.interNodePairs, dtype=np.int64)

        if verbose > 0:
            msgStr = "{0} pairs of nodes connecting the two selections."
            print(msgStr.format(len(self.interNodePairs)))

        self.contactNodesInter = np.unique(self.interNodePairs)

        if verbose > 0:
            msgStr = "{0} unique nodes in interface node pairs."
            print(msgStr.format(len(self.contactNodesInter)))

        return len(self.contactNodesInter)
