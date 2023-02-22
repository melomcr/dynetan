import pytest
from MDAnalysis import lib


class TestToolkitClass:

    def test_diag_func(self):
        from dynetan.toolkit import diagnostic

        assert diagnostic() == lib.distances.USED_OPENMP

    def test_get_lin_index_c(self):
        from dynetan.toolkit import getLinIndexC

        testsList = [
            (0, 5, 10, 4),
            (1, 5, 10, 12)
        ]

        for src, trg, dim, res in testsList:
            testRes = getLinIndexC(src, trg, dim)

            assert testRes == res

    def test_node_selection_conversions(self, dnap_omp):
        from dynetan.toolkit import getNGLSelFromNode
        from dynetan.toolkit import getNodeFromSel
        from dynetan.toolkit import getSelFromNode

        dnap_omp.checkSystem()
        dnap_omp.selectSystem(withSolvent=False)
        dnap_omp.prepareNetwork()

        # Test node from MDanalysis selection
        ref_str = 'segid ENZY and resname GLU and resid 115'
        test_node = getNodeFromSel(ref_str,
                                   dnap_omp.nodesAtmSel,
                                   dnap_omp.atomToNode)
        assert test_node == 104

        ref_str = 'segid OMP'
        test_node = getNodeFromSel(ref_str,
                                   dnap_omp.nodesAtmSel,
                                   dnap_omp.atomToNode)
        assert list(test_node) == [215, 216]

        # Test creation of MDanalysis selection string from node
        ref_str = 'resname GLU and resid 115 and segid ENZY and name CA'
        test_str = getSelFromNode(nodeIndx=104,
                                  atomsel=dnap_omp.nodesAtmSel,
                                  atom=True)
        assert ref_str == test_str

        ref_str = 'resname GLU and resid 115 and segid ENZY'
        test_str = getSelFromNode(nodeIndx=104,
                                  atomsel=dnap_omp.nodesAtmSel,
                                  atom=False)
        assert ref_str == test_str

        # Test creation of NGL selection
        ref_str = '115 and GLU and .CA'
        test_str = getNGLSelFromNode(nodeIndx=104,
                                     atomsel=dnap_omp.nodesAtmSel,
                                     atom=True)
        assert ref_str == test_str

        ref_str = '115 and GLU'
        test_str = getNGLSelFromNode(nodeIndx=104,
                                     atomsel=dnap_omp.nodesAtmSel,
                                     atom=False)
        assert ref_str == test_str

    def test_format_node_grp(self, dnap_omp, capsys):

        from dynetan.toolkit import formatNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        formatNodeGroups(atm_group, ["CA"])

        # Check verbosity output for multicore run
        captured = capsys.readouterr()

        assert 'usrNodeGroups["VAL"] = {}' in captured.out
        assert 'usrNodeGroups["VAL"]["CA"] = ' in captured.out

    def test_format_node_grp_2(self, dnap_omp, capsys):

        from dynetan.toolkit import formatNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        nodes = ["CA", "CB"]
        groups = [['C', 'O', 'N', 'CA'], ['CB', 'CG1', 'CG2']]
        formatNodeGroups(atm_group, nodes, groups)

        # Check verbosity output for multicore run
        captured = capsys.readouterr()

        assert 'usrNodeGroups["VAL"] = {}' in captured.out
        assert 'usrNodeGroups["VAL"]["CA"] = ' in captured.out
        assert 'usrNodeGroups["VAL"]["CB"]' in captured.out

    @pytest.mark.xfail
    def test_format_node_grp_missing_node(self, dnap_omp):

        from dynetan.toolkit import formatNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        formatNodeGroups(atm_group, ["X"])

    @pytest.mark.xfail
    def test_format_node_grp_missing_arg(self, dnap_omp):
        from dynetan.toolkit import formatNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        formatNodeGroups(atm_group, ["CA", "CB"])

    @pytest.mark.xfail
    def test_format_node_grp_num_groups(self, dnap_omp):
        from dynetan.toolkit import formatNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        formatNodeGroups(atm_group, ["CA", "CB"], [['C', 'O', 'N', 'CA']])

    node_group_1 = {"VAL": {}}
    node_group_1["VAL"]["CA"] = {'O', 'CB', 'CA', 'CG1', 'C', 'CG2', 'N'}

    node_group_2 = {"VAL": {}}
    node_group_2["VAL"]["CA"] = {'C', 'N', 'CA', 'O'}
    node_group_2["VAL"]["CB"] = {'CG2', 'CB', 'CG1'}

    node_group_3 = {"VAL": {}}
    node_group_3["VAL"]["CA"] = {'C', 'CA', 'O'}
    node_group_3["VAL"]["CB"] = {'CG2', 'CB', 'CG1'}

    @pytest.mark.parametrize(
        ("node_group", "node_sel"), [
            pytest.param(node_group_1, ""),
            pytest.param(node_group_2, "CB"),
            pytest.param(node_group_3, ""),
        ])
    def test_nglview(self, dnap_omp, node_group, node_sel):
        import nglview as nv
        from dynetan.toolkit import showNodeGroups

        sel_str = "resid 11 and resname VAL and (not (name H* or name [123]H*))"
        atm_group = dnap_omp.workU.select_atoms(sel_str)

        w = nv.show_mdanalysis(atm_group)

        showNodeGroups(w, atm_group, node_group, node_sel)
