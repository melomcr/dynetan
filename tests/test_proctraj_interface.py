import pytest

from .test_proctraj_cartesian import load_sys_solv_mode

ref_pairs_list = [[  7, 215], [  9, 215], [ 31, 215], [ 31, 216],  # noqa: E201
                  [ 59, 216], [ 61, 216], [ 85, 216], [112, 216],  # noqa: E201
                  [114, 216], [115, 216], [116, 216], [144, 216],
                  [169, 215], [169, 216], [171, 215], [173, 215],
                  [190, 215], [191, 215], [192, 215]]
ref_nodes_list = [7,   9,  31,  59,  61,  85, 112, 114, 115, 116, 144, 169, 171,
                  173, 190, 191, 192, 215, 216]


@pytest.mark.parametrize(
    ("selections", "dist", "ref_contacts", "ref_pairs", "ref_nodes"), [
        pytest.param(("segid ENZY", "segid OMP"), 15,
                     19, ref_pairs_list, ref_nodes_list),
        pytest.param(("segid ENZY and resname GLU and resid 115", "segid OMP"),
                     10, 0, [], [])
    ])
def test_interface_analysis(dnap_omp, capsys, selections, dist,
                            ref_contacts, ref_pairs, ref_nodes):
    dnap = load_sys_solv_mode(dnap_omp, False, "all")

    contact_nodes_inter = dnap.interfaceAnalysis(selAstr=selections[0],
                                                 selBstr=selections[1],
                                                 betweenDist=dist,
                                                 verbose=2)

    # Check verbosity output
    captured = capsys.readouterr()

    assert contact_nodes_inter == ref_contacts

    if ref_contacts == 0:
        assert "No contacts found in timestep" in captured.out
        assert "Check your selections and sampling" in captured.out

    else:
        assert "pairs of nodes connecting the two selections." in captured.out
        assert "unique nodes in interface node pairs." in captured.out

        for pair in ref_pairs:
            assert pair in dnap.interNodePairs

        for node in ref_nodes:
            assert node in dnap.contactNodesInter
