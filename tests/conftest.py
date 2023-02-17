import pytest
import os
import dynetan as dna

######
psf_fn_omp = "decarboxylase.0.psf"
dcd_fn_omp = "decarboxylase.1.short.dcd"
######


@pytest.fixture(scope="session")
def test_data_dir():
    from . import test_data

    from importlib.resources import path as pkg_path

    # Get path to trajectory and topology files:
    with pkg_path(test_data, psf_fn_omp) as path_to_test_data_gen:
        path_to_test_data = path_to_test_data_gen.parent.resolve()

    return path_to_test_data


@pytest.fixture
def dnap_omp(test_data_dir):
    psfFile = os.path.join(test_data_dir, psf_fn_omp)
    dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

    dnap_omp = dna.proctraj.DNAproc(notebookMode=False)

    # Test the default wrapper for loading a list of paths to DCDs
    dnap_omp.loadSystem(psfFile, dcdFiles)

    # Initialize user-defined residues
    node_grps = {"OMP": {}, "TIP3": {}}

    # Define nodes and atom groups for the ligand
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

    # Define node and atom group for the solvent
    node_grps["TIP3"]["OH2"] = set("OH2 H1 H2".split())

    dnap_omp.setNumWinds(2)
    dnap_omp.setNumSampledFrames(2)
    dnap_omp.setSolvNames(["TIP3"])
    dnap_omp.setSegIDs(["OMP", "ENZY"])

    dnap_omp.setNodeGroups(node_grps)

    return dnap_omp


@pytest.fixture(scope="session")
def tmp_dna_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traj_data")


@pytest.fixture
def dnap_omp_loaded(dnap_omp):
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=0)

    dnap_omp.alignTraj()

    dnap_omp.findContacts(stride=2, verbose=0)

    dnap_omp.filterContacts(notSameRes=True,
                            notConsecutiveRes=True,
                            removeIsolatedNodes=True,
                            verbose=0)

    return dnap_omp
