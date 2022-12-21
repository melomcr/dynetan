import pytest
import dynetan as dna
import MDAnalysis as mda

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
def dnap_omp(test_data_dir):
    psfFile = os.path.join(test_data_dir, psf_fn_omp)
    dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

    dnap_omp = dna.proctraj.DNAproc(notebookMode=False)

    # Test the default wrapper for loading a list of paths to DCDs
    dnap_omp.loadSystem(psfFile, dcdFiles)

    node_grps = {"OMP": {}}
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

    dnap_omp.setNumWinds(2)
    dnap_omp.setNumSampledFrames(2)
    dnap_omp.setSolvNames(["TIP3"])
    dnap_omp.setSegIDs(["OMP", "ENZY"])

    dnap_omp.setNodeGroups(node_grps)

    return dnap_omp


@pytest.fixture(scope="session")
def tmp_dna_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traj_data")


class TestPackage:

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

    @pytest.mark.xfail(raises=AssertionError)
    def test_exist_universe(self):
        dnap = dna.proctraj.DNAproc(notebookMode=False)
        dnap.checkSystem()

    def test_check_system(self, dnap_omp):
        dnap_omp.checkSystem()

        expect_all_res = {'ASN', 'MET', 'LEU', 'CYS', 'SOD', 'LYS', 'OMP',
                          'VAL', 'GLU', 'TIP3', 'GLY', 'ALA', 'ASP', 'ILE',
                          'PRO', 'SER', 'THR', 'PHE', 'HSD', 'ARG', 'TYR', 'GLN'}

        assert dnap_omp.allResNamesSet == expect_all_res

        expect_sel_names = {'ASN', 'MET', 'LEU', 'CYS', 'LYS', 'OMP', 'VAL',
                            'GLU', 'GLY', 'ALA', 'ASP', 'ILE', 'PRO', 'SER',
                            'THR', 'PHE', 'HSD', 'ARG', 'TYR', 'GLN'}

        assert dnap_omp.selResNamesSet == expect_sel_names

        expect_not_sel_names = {'TIP3', 'SOD'}

        assert dnap_omp.notSelResNamesSet == expect_not_sel_names

        expect_not_sel_segid = {'WT5', 'WT4', 'WT8', 'WT3', 'WT7', 'ION', 'WT1',
                                'WT2', 'WT6'}

        assert dnap_omp.notSelSegidSet == expect_not_sel_segid

    @pytest.mark.parametrize(("solv", "sel", "verb"), [
        pytest.param(0, "all", 0, marks=pytest.mark.xfail),
        pytest.param(True, None, 0, marks=pytest.mark.xfail),
        pytest.param(True, "all", False, marks=pytest.mark.xfail)])
    def test_select_system(self, dnap_omp, solv, sel, verb):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(with_solvent=solv,
                              input_sel_str=sel,
                              verbose=verb)

    @pytest.mark.xfail()
    def test_select_system_condition(self, test_data_dir):

        psfFile = os.path.join(test_data_dir, psf_fn_omp)
        dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

        dnap_fail = dna.proctraj.DNAproc(notebookMode=False)

        # Test the default wrapper for loading a list of paths to DCDs
        dnap_fail.loadSystem(psfFile, dcdFiles)

        node_grps = {"OMP": {}}
        node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
        node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

        dnap_fail.setNumWinds(2)
        dnap_fail.setNumSampledFrames(2)

        dnap_fail.setSolvNames([])  # EMPTY LIST TO CAUSE FAILED INPUT TEST

        dnap_fail.setSegIDs(["OMP", "ENZY"])

        dnap_fail.setNodeGroups(node_grps)

        dnap_fail.checkSystem()
        dnap_fail.selectSystem(with_solvent=False, input_sel_str="")

    @pytest.mark.parametrize(("solv", "sel", "verb", "n_atoms"), [
        (True, "protein", 0, 3291),
        (True, "protein and (not (name H* or name [123]H*))", 0, 1635),
        (True, "", 0, 1928),
        (False, "", 0, 1659)
    ])
    def test_select_system_proc(self, dnap_omp, solv, sel, verb, n_atoms):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(with_solvent=solv,
                              input_sel_str=sel,
                              verbose=verb)
        assert len(dnap_omp.getU().atoms) == n_atoms

    @pytest.mark.parametrize(("verb", "out_txt"), [
        (1, "New residue types included"),
        (2, "New residues included")])
    def test_select_system_verb(self, dnap_omp, capfd, verb, out_txt):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(with_solvent=True,
                              input_sel_str="",
                              verbose=verb)

        captured = capfd.readouterr()
        assert out_txt in captured.out

    @pytest.mark.parametrize(("solv", "pre_sel", "n_atoms"), [
        (True, "protein", 1635),
        (True, "not resname TIP3", 1659),
        (False, "protein", 1635),
        (False, "not resname TIP3", 1659)])
    def test_select_system_sel(self, dnap_omp, solv, pre_sel, n_atoms):

        # We artificially change the test system to check class methods.

        from MDAnalysis.analysis.base import AnalysisFromFunction as mdaAFF
        from MDAnalysis.coordinates.memory import MemoryReader as mdaMemRead
        import numpy as np

        test_sel = dnap_omp.getU().select_atoms(pre_sel)
        dnap_omp.workU = mda.core.universe.Merge(test_sel)

        resObj = mdaAFF(lambda ag: ag.positions.copy(), test_sel).run().results

        # This checks the type of the MDAnalysis results. Prior to version 2.0.0,
        # MDA returned a numpy.ndarray with the trajectory coordinates. After
        # version 2.0.0, it returns a results object that contains the trajectory.
        # With this check, the code can handle both APIs.
        if not isinstance(resObj, np.ndarray):
            resObj = resObj['timeseries']

        dnap_omp.workU.load_new(resObj, format=mdaMemRead)

        # END of artificial system changes

        dnap_omp.checkSystem()
        dnap_omp.selectSystem(with_solvent=solv, input_sel_str="")
        assert len(dnap_omp.getU().atoms) == n_atoms