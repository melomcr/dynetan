import pytest
import dynetan as dna
import MDAnalysis as mda

import os

from .conftest import psf_fn_omp, dcd_fn_omp


class TestPackageSysVerificationSelection:
    """
    This class aggregates functions to test the system selection and verification
    methods implemented in DyNetAn.
    """

    xfail_strict = pytest.mark.xfail(strict=True)

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

    @pytest.mark.xfail(raises=AssertionError, strict=True)
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
        pytest.param(0, "all", 0, marks=xfail_strict),
        pytest.param(True, None, 0, marks=xfail_strict),
        pytest.param(True, "all", "False", marks=xfail_strict)],)
    def test_select_system(self, dnap_omp, solv, sel, verb):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(withSolvent=solv, inputSelStr=sel, verbose=verb)

    def test_select_system_pre_check(self, dnap_omp):
        # We will intentionally SKIP the "checkSystem" method to test if
        # the "selectSystem" method will automatically call "checkSystem" to
        # initialize variables.
        dnap_omp.selectSystem(withSolvent=False, inputSelStr="", verbose=0)

        assert dnap_omp.notSelSegidSet is not None

    @pytest.mark.xfail(strict=True)
    def test_select_system_condition(self, test_data_dir):

        psfFile = os.path.join(test_data_dir, psf_fn_omp)
        dcdFiles = [os.path.join(test_data_dir, dcd_fn_omp)]

        dnap_fail = dna.proctraj.DNAproc(notebookMode=False)

        # Test the default wrapper for loading a list of paths to DCDs
        dnap_fail.loadSystem(psfFile, dcdFiles)

        node_grps = {"OMP": {}}
        node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
        node_grps["OMP"]["P"] = set("P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' "
                                    "O2' O3'".split())

        dnap_fail.setNumWinds(2)
        dnap_fail.setNumSampledFrames(2)

        dnap_fail.setSolvNames([])  # EMPTY LIST TO CAUSE FAILED INPUT TEST

        dnap_fail.setSegIDs(["OMP", "ENZY"])

        dnap_fail.setNodeGroups(node_grps)

        dnap_fail.checkSystem()
        dnap_fail.selectSystem(withSolvent=False, inputSelStr="")

    @pytest.mark.parametrize(("solv", "sel", "verb", "n_atoms"), [
        (True, "protein", 0, 3291),
        (True, "protein and (not (name H* or name [123]H*))", 0, 1635),
        (True, "", 0, 1928),
        (False, "", 0, 1659)
    ])
    def test_select_system_proc(self, dnap_omp, solv, sel, verb, n_atoms):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(withSolvent=solv, inputSelStr=sel, verbose=verb)
        assert len(dnap_omp.getU().atoms) == n_atoms

    @pytest.mark.parametrize(("verb", "out_txt"), [
        (1, "New residue types included"),
        (2, "New residues included")])
    def test_select_system_verb(self, dnap_omp, capfd, verb, out_txt):
        dnap_omp.checkSystem()
        dnap_omp.selectSystem(withSolvent=True, inputSelStr="", verbose=verb)

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
        dnap_omp.selectSystem(withSolvent=solv, inputSelStr="")
        assert len(dnap_omp.getU().atoms) == n_atoms
