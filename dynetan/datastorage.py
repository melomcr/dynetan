#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: melomcr

import numpy as np
import pickle, h5py

class DNAdata:
    '''Data storage and management class.
    
    The Dynamic Network Analysis data class contains the 
    infrastructure to save all data required to analyze and reproduce
    a DNA study of a biomolecular system.
    
    The essential network and correlation coefficients data is stored in 
    an HDF5 file using the H5Py module, allowing long term storage.
    The remaining data is stored in NumPy binary format and the
    NetworkX graph objects are stores in a pickle format. This is not 
    intended for term storage, but the data can be easily recovered from
    the HDF5 data.
    
    '''
    
    def __init__(self):
        '''Constructor
        
        Sets initial values for class variables.
        
        '''
        
        self.contactMat         = None
        
        self.atomToNode         = None
        
        self.nodesIxArray       = None
        
        self.numNodes           = None
        
        self.nodeDists          = None
        
        self.corrMatAll         = None
        
        self.distsAll           = None
        
        self.preds              = None
        
        self.maxDist            = None
        
        self.maxDirectDist      = None
        
        self.btws               = None
        
        self.interNodePairs     = None
        
        self.contactNodesInter  = None
        
        self.nxGraphs           = None
        
        self.nodesComm          = None
        
        # The following attributes are not saved to file, 
        #  they are reconstructed from the loaded information.
        self.nodesAtmSel        = None
        self.numWinds           = None
        
    def saveToFile(self, fileNameRoot):
        '''Function that saves all the data stored in a DNAdata object.
        
        Args:
            fileNameRoot (str): Root of the multiple data files to be writen.
        
        '''
        
        # Opens the HDF5 file and store all data.
        with h5py.File(fileNameRoot + ".hf", "w") as f:

            f_contactMat = f.create_dataset("contactMat", 
                                            shape= self.contactMat.shape, 
                                            dtype= self.contactMat.dtype,
                                            data= self.contactMat)

            f_atomToNode = f.create_dataset("atomToNode", 
                                            shape= self.atomToNode.shape, 
                                            dtype= self.atomToNode.dtype,
                                            data= self.atomToNode)

            f_nodesIxArray = f.create_dataset("nodesIxArray", 
                                            shape= self.nodesIxArray.shape, 
                                            dtype= self.nodesIxArray.dtype,
                                            data= self.nodesIxArray)

            f_numNodes = f.create_dataset("numNodes", dtype='i', data= self.numNodes)

            f_nodeDists = f.create_dataset("nodeDists", 
                                            shape= self.nodeDists.shape, 
                                            dtype= self.nodeDists.dtype,
                                            data= self.nodeDists)

            f_corrMatAll = f.create_dataset("corrMatAll", 
                                            shape= self.corrMatAll.shape, 
                                            dtype= self.corrMatAll.dtype,
                                            data= self.corrMatAll)

            f_distsAll = f.create_dataset("distsAll", 
                                            shape= self.distsAll.shape, 
                                            dtype= self.distsAll.dtype,
                                            data= self.distsAll)

            f_maxDist = f.create_dataset("maxDist", dtype='f8', data= self.maxDist)
            
            f_maxDirectDist = f.create_dataset("maxDirectDist", dtype='f8', data= self.maxDirectDist ) 

            f_interNodePairs = f.create_dataset("interNodePairs", 
                                            shape= self.interNodePairs.shape, 
                                            dtype= self.interNodePairs.dtype,
                                            data= self.interNodePairs)

            f_contactNodesInter = f.create_dataset("contactNodesInter", 
                                            shape= self.contactNodesInter.shape, 
                                            dtype= self.contactNodesInter.dtype,
                                            data= self.contactNodesInter)
        
        with open(fileNameRoot + "_nxGraphs.pickle", 'wb') as outfile:
            pickle.dump(self.nxGraphs, outfile)
        
        np.save(fileNameRoot + "_preds.npy", self.preds)

        np.save(fileNameRoot + "_btws.npy", self.btws)

        np.save(fileNameRoot + "_nodesComm.npy", self.nodesComm)
        
        
    def loadFromFile(self, fileNameRoot):
        '''Function that loads all the data stored in a DNAdata object.
        
        Args:
            fileNameRoot (str): Root of the multiple data files to be loaded.
        
        '''
        
        with h5py.File(fileNameRoot + ".hf", "r") as f:
            
            for key in f.keys():
                print( key, f[key].dtype, f[key].shape, f[key].size)
                
                if f[key].size > 1:
                    # Stores value in object
                    setattr(self, key, np.zeros( f[key].shape, dtype=f[key].dtype))
                    
                    f[key].read_direct( getattr(self, key) )
                else:
                    # For a *scalar* H5Py Dataset, we index using an empty souple.
                    setattr(self, key, f[key][()])
        
        with open(fileNameRoot + "_nxGraphs.pickle", 'rb') as infile:
            self.nxGraphs = pickle.load(infile)
        
        self.preds = np.load(fileNameRoot + "_preds.npy", allow_pickle = True).item()
        
        self.btws = np.load(fileNameRoot + "_btws.npy", allow_pickle = True).item()
        
        self.nodesComm = np.load(fileNameRoot + "_nodesComm.npy", allow_pickle = True).item()
        
        ### Load derived attributes
        
        self.numWinds       = self.corrMatAll.shape[0]

