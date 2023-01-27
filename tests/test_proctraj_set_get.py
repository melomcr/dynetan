import os.path

import pytest

import dynetan as dna
from dynetan import proctraj as dnapt
from dynetan import contact as ct

from .conftest import psf_fn_omp, dcd_fn_omp


@pytest.fixture
def dnap():
    return dnapt.DNAproc(notebookMode=False)


class TestProcTraj:

    def test_version(self):

        assert hasattr(dna, "__version__")

        assert dna.version.__version__ == dna.__version__

    def test_prog_bar(self):
        from tqdm.notebook import tqdm_notebook as tqdm_nb
        from tqdm import tqdm as tqdm_cli

        dnap_progBar = dnapt.DNAproc(notebookMode=True)
        assert dnap_progBar.progBar == tqdm_nb
        assert dnap_progBar.asciiMode is False

        dnap_progBar = dnapt.DNAproc(notebookMode=False)
        assert dnap_progBar.progBar == tqdm_cli
        assert dnap_progBar.asciiMode is True

    @pytest.mark.parametrize("num_winds", [
        1,
        12345,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param("10", marks=pytest.mark.xfail)])
    def test_set_num_winds(self, dnap, num_winds):
        dnap.setNumWinds(num_winds)
        assert dnap.numWinds == num_winds

    @pytest.mark.parametrize("num_sf", [
        1,
        12345,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param("10", marks=pytest.mark.xfail)])
    def test_set_num_sampled_frames(self, dnap, num_sf):
        dnap.setNumSampledFrames(num_sf)
        assert dnap.numSampledFrames == num_sf

    @pytest.mark.parametrize("cutoff_dist", [
        4.5,
        5,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param("10", marks=pytest.mark.xfail)])
    def test_set_cutoff_dist(self, dnap, cutoff_dist):
        dnap.setCutoffDist(cutoff_dist)
        assert dnap.cutoffDist == cutoff_dist

    @pytest.mark.parametrize("cntc_perst", [
        0.75,
        0.1,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(1.2, marks=pytest.mark.xfail),
        pytest.param("10", marks=pytest.mark.xfail)])
    def test_set_cntc_perst(self, dnap, cntc_perst):
        dnap.setContactPersistence(cntc_perst)
        assert dnap.contactPersistence == cntc_perst

    @pytest.mark.parametrize("solvs", [
        ["TIP3"],
        ["TIP3", "CHOL"],
        pytest.param(["TIP3", 0], marks=pytest.mark.xfail),
        pytest.param(0.5, marks=pytest.mark.xfail),
        pytest.param("TIP3", marks=pytest.mark.xfail)])
    def test_set_solvs(self, dnap, solvs):
        dnap.setSolvNames(solvs)
        assert dnap.solventNames == solvs

        dnap.seth2oName(solvs)
        assert dnap.solventNames == solvs

    @pytest.mark.parametrize("ids", [
        ["PROT"],
        ["PROT", "PEP"],
        pytest.param(["PROT", 0], marks=pytest.mark.xfail),
        pytest.param(0.5, marks=pytest.mark.xfail),
        pytest.param("PROT", marks=pytest.mark.xfail)])
    def test_set_ids(self, dnap, ids):
        dnap.setSegIDs(ids)
        assert dnap.segIDs == ids

    node_groups = {}
    node_groups["TIP3"] = {}
    node_groups["TIP3"]["OH2"] = set("OH2 H1 H2".split())
    node_groups["OMP"] = {}
    node_groups["OMP"]["N1"] = set("N1 C2 O2 N3".split())
    node_groups["OMP"]["P"] = set("P OP1 OP2 OP3 O5' C5' C4'".split())

    @pytest.mark.parametrize("ngrps", [
        node_groups,
        pytest.param({"TIP3": 12345}, marks=pytest.mark.xfail),
        pytest.param({"TIP3": "OH2"}, marks=pytest.mark.xfail),
        pytest.param({"TIP3": {"OH2": ["a", "b"]}}, marks=pytest.mark.xfail)])
    def test_set_groups(self, dnap, ngrps):
        dnap.setNodeGroups(ngrps)

    @pytest.mark.parametrize("ngrps", [
        pytest.param(node_groups, marks=pytest.mark.xfail)])
    def test_set_usr_groups_deprecation(self, dnap, ngrps):
        dnap.setUsrNodeGroups(ngrps)

    @pytest.mark.parametrize("ngrps", [
        pytest.param(node_groups, marks=pytest.mark.xfail)])
    def test_set_groups_deprecation(self, dnap, ngrps):
        dnap.setCustomResNodes(ngrps)

    @pytest.mark.parametrize(("mode_str", "mode_int"), [
            ("all", ct.MODE_ALL),
            ("capped", ct.MODE_CAPPED),
            pytest.param("all", ct.MODE_CAPPED, marks=pytest.mark.xfail),
            pytest.param("any", ct.MODE_ALL, marks=pytest.mark.xfail)])
    def test_set_dist_mode(self, dnap, mode_str, mode_int):
        dnap.setDistanceMode(mode_str)
        assert dnap.distanceMode == mode_int

    @pytest.mark.parametrize(("str_fn", "traj_fns"), [
        (psf_fn_omp, dcd_fn_omp),
        pytest.param(psf_fn_omp, "no_file_here", marks=pytest.mark.xfail)])
    def test_load_system(self, dnap, test_data_dir, str_fn, traj_fns):
        """Test loading system files into MDAnalysis

        This tests the default wrapper for loading a structure file and a
        trajectory file into an MDA universe.

        Args:
            dnap: Dynetan DNAproc object
            test_data_dir: Path to where test data is stored.
            str_fn: Path to structure file
            traj_fns: Path to trajectory file

        Returns:
            None
        """
        struc_fn = os.path.join(test_data_dir, str_fn)
        traj_fn = os.path.join(test_data_dir, traj_fns)
        dnap.loadSystem(struc_fn, traj_fn)
