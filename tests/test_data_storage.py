import os
from dynetan import DNAdata

from .test_proctraj_cartesian import load_sys_solv_mode


def test_data_storage(tmp_path, dnap_omp):

    tmpdir = tmp_path
    print("Using tmpdir:", tmpdir)

    path_root = os.path.join(tmpdir, "test_dnaData")

    # Prepare Data
    dnap = load_sys_solv_mode(dnap_omp, False, "all")
    dnap.calcCartesian(backend="serial", verbose=0)
    dnap.calcGraphInfo()
    dnap.calcOptPaths(ncores=1)
    dnap.calcBetween(ncores=1)
    dnap.calcEigenCentral()
    dnap.calcCommunities()

    selections = ("segid ENZY", "segid OMP")
    dnap.interfaceAnalysis(selAstr=selections[0],
                           selBstr=selections[1],
                           betweenDist=15,
                           verbose=0)

    # Save Data
    dnap.saveData(file_name_root=path_root)

    import warnings
    # suppress some MDAnalysis warnings about writing DCD files
    warnings.filterwarnings('ignore')
    dnap.saveReducedTraj(file_name_root=path_root, stride=1)

    # Check if each expected file exists
    assert os.path.exists(path_root + "_btws.npy")
    assert os.path.exists(path_root + "_nodesComm.npy")
    assert os.path.exists(path_root + "_nxGraphs.pickle")
    assert os.path.exists(path_root + "_preds.npy")
    assert os.path.exists(path_root + "_reducedTraj.dcd")
    assert os.path.exists(path_root + "_reducedTraj.pdb")
    assert os.path.exists(path_root + ".hf")

    # Load Data
    dna_data = DNAdata()
    dna_data.loadFromFile(file_name_root=path_root)

    # Check if loaded object has all expected data
    assert dna_data.numWinds == 2
    assert dna_data.contactMat.shape == (217, 217)
    assert dna_data.atomToNode.shape == (1659,)
    assert dna_data.nodesIxArray.shape == (217,)
    assert dna_data.numNodes == 217

    assert dna_data.nodeDists.shape == (4, 23436)
    assert dna_data.corrMatAll.shape == (2, 217, 217)
    assert dna_data.distsAll.shape == (2, 217, 217)
    assert round(dna_data.maxDist, 5) == 25.69056
    assert round(dna_data.maxDirectDist, 5) == 4.11881

    assert dna_data.interNodePairs.shape == (19, 2)
    assert dna_data.contactNodesInter.shape == (19,)
