import pytest
import dynetan as dna

import os

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
def dnap_omp_loaded(test_data_dir):
    psfFile = os.path.join(test_data_dir, psf_fn_omp)
    dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

    dnap_omp = dna.proctraj.DNAproc(notebookMode=False)

    # Test the default wrapper for loading a list of paths to DCDs
    dnap_omp.loadSystem(psfFile, dcdFiles)

    return dnap_omp


@pytest.fixture(scope="session")
def tmp_dna_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traj_data")


class TestPackage:
    def test_version(self):

        assert hasattr(dna, "__version__")

        assert dna.version.__version__ == dna.__version__

    def test_load_data(self, test_data_dir):
        psfFile  = os.path.join(test_data_dir, psf_fn_omp)
        dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

        dnap = dna.proctraj.DNAproc(notebookMode=False)

        # Test the default wrapper for loading a list of paths to DCDs
        dnap.loadSystem(psfFile, dcdFiles)
        assert len(dnap.getU().trajectory) == 20

        # Test the default wrapper for loading a single DCD path
        dnap.loadSystem(psfFile, dcdFiles[0])
        assert len(dnap.getU().trajectory) == 20

    def test_get_node_from_sel(self, dnap_omp_loaded):
        pass

    def test_get_ngl_sel_from_node(self, dnap_omp_loaded):
        pass
