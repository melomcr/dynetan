#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: melomcr

import numpy as np
import numpy.typing as npt
import pickle
import h5py
from typing import Any, Union
import MDAnalysis as mda  # noqa


class DNAdata:
    """Data storage and management class.

    The Dynamic Network Analysis data class contains the
    infrastructure to save all data required to analyze and reproduce
    a DNA study of a biomolecular system.

    The essential network and correlation coefficients data is stored in
    an HDF5 file using the H5Py module, allowing long term storage.
    The remaining data is stored in NumPy binary format and the
    NetworkX graph objects are stores in a pickle format. This is not
    intended for term storage, but the data can be easily recovered from
    the HDF5 data.

    """

    nxGraphs: Any
    preds: Any
    btws: Any
    nodesComm: Any

    def __init__(self) -> None:
        """Constructor

        Sets initial values for class variables.

        """

        self.contactMat: npt.NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self.atomToNode: npt.NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self.nodesIxArray: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self.numNodes: int = 0
        self.nodeDists: Union[npt.NDArray[np.float64], None] = None
        self.corrMatAll: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self.distsAll: npt.NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        self.preds: Any
        self.maxDist: float = 0.0
        self.maxDirectDist: float = 0.0
        self.btws: Any
        self.interNodePairs: Union[npt.NDArray[np.int64], None] = None
        self.contactNodesInter: Union[npt.NDArray[np.int64], None] = None
        self.nxGraphs: Any
        self.nodesComm: Any

        # The following attributes are not saved to file,
        #  they are reconstructed from the loaded information.
        self.nodesAtmSel: mda.AtomGroup
        self.numWinds: int = 0

    def saveToFile(self, file_name_root: str) -> None:
        """Function that saves all the data stored in a DNAdata object.

        Args:
            file_name_root (str): Root of the multiple data files to be writen.

        """

        # Opens the HDF5 file and store all data.
        with h5py.File(file_name_root + ".hf", "w") as f:

            # f_contactMat
            f.create_dataset("contactMat",
                             shape=self.contactMat.shape,
                             dtype=self.contactMat.dtype,
                             data=self.contactMat)

            # f_atomToNode
            f.create_dataset("atomToNode",
                             shape=self.atomToNode.shape,
                             dtype=self.atomToNode.dtype,
                             data=self.atomToNode)

            # f_nodesIxArray
            f.create_dataset("nodesIxArray",
                             shape=self.nodesIxArray.shape,
                             dtype=self.nodesIxArray.dtype,
                             data=self.nodesIxArray)

            # f_numNodes
            f.create_dataset("numNodes", dtype='i', data=self.numNodes)

            if self.nodeDists is not None:
                # f_nodeDists
                f.create_dataset("nodeDists",
                                 shape=self.nodeDists.shape,
                                 dtype=self.nodeDists.dtype,
                                 data=self.nodeDists)

            # f_corrMatAll
            f.create_dataset("corrMatAll",
                             shape=self.corrMatAll.shape,
                             dtype=self.corrMatAll.dtype,
                             data=self.corrMatAll)

            # f_distsAll
            f.create_dataset("distsAll",
                             shape=self.distsAll.shape,
                             dtype=self.distsAll.dtype,
                             data=self.distsAll)

            # f_maxDist
            f.create_dataset("maxDist", dtype='f8', data=self.maxDist)

            # f_maxDirectDist
            f.create_dataset("maxDirectDist",
                             dtype='f8',
                             data=self.maxDirectDist)

            if self.interNodePairs is not None:
                # f_interNodePairs
                f.create_dataset("interNodePairs",
                                 shape=self.interNodePairs.shape,
                                 dtype=self.interNodePairs.dtype,
                                 data=self.interNodePairs)

            if self.contactNodesInter is not None:
                # f_contactNodesInter
                f.create_dataset("contactNodesInter",
                                 shape=self.contactNodesInter.shape,
                                 dtype=self.contactNodesInter.dtype,
                                 data=self.contactNodesInter)

        with open(file_name_root + "_nxGraphs.pickle", 'wb') as outfile:
            pickle.dump(self.nxGraphs, outfile)

        np.save(file_name_root + "_preds.npy", self.preds)

        np.save(file_name_root + "_btws.npy", self.btws)

        np.save(file_name_root + "_nodesComm.npy", self.nodesComm)

    def loadFromFile(self, file_name_root: str) -> None:
        """Function that loads all the data stored in a DNAdata object.

        Args:
            file_name_root (str): Root of the multiple data files to be loaded.

        """

        with h5py.File(file_name_root + ".hf", "r") as f:

            for key in f.keys():
                print(key, f[key].dtype, f[key].shape, f[key].size)

                if f[key].size > 1:
                    # Stores value in object
                    setattr(self, key, np.zeros(f[key].shape, dtype=f[key].dtype))

                    f[key].read_direct(getattr(self, key))
                else:
                    # For a *scalar* H5Py Dataset, we index using an empty tuple.
                    setattr(self, key, f[key][()])

        with open(file_name_root + "_nxGraphs.pickle", 'rb') as infile:
            self.nxGraphs = pickle.load(infile)

        self.preds = np.load(file_name_root + "_preds.npy",
                             allow_pickle=True).item()

        self.btws = np.load(file_name_root + "_btws.npy",
                            allow_pickle=True).item()

        self.nodesComm = np.load(file_name_root + "_nodesComm.npy",
                                 allow_pickle=True).item()

        # Load derived attributes

        self.numWinds = self.corrMatAll.shape[0]
