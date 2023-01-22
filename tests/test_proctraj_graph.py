import pytest
import networkx as nx

from .test_proctraj_checksys_selectsys import test_data_dir  # NOQA - PyCharm
from .test_proctraj_checksys_selectsys import dnap_omp  # NOQA - PyCharm
from .test_proctraj_corr import dnap_omp_loaded  # NOQA - PyCharm
from .test_proctraj_cartesian import load_sys_solv_mode # NOQA - PyCharm


def test_calc_graph(dnap_omp):
    # NumNodes, NumEdges, Density, Transitivity, resname, resid
    ref_graph_data = [(217, 516, 0.022, 0.1638, "LEU", 122),
                      (217, 463, 0.0198, 0.1687, "TYR", 154)]

    from operator import itemgetter
    from dynetan.toolkit import getSelFromNode

    dnap = load_sys_solv_mode(dnap_omp, False, "all")

    dnap.calcCartesian(backend="serial", verbose=0)

    dnap.calcGraphInfo()

    for wind_i in range(dnap.numWinds):
        assert len(dnap.nxGraphs[wind_i].nodes) == ref_graph_data[wind_i][0]
        assert len(dnap.nxGraphs[wind_i].edges) == ref_graph_data[wind_i][1]

        density = round(nx.density(dnap.nxGraphs[wind_i]), 4)
        assert density == ref_graph_data[wind_i][2]

        transitivity = round(nx.transitivity(dnap.nxGraphs[wind_i]), 4)
        assert transitivity == ref_graph_data[wind_i][3]

        sorted_degree = sorted(dnap.getDegreeDict(wind_i).items(),
                               key=itemgetter(1), reverse=True)

        node, degree = sorted_degree[0]

        sel_str = getSelFromNode(node, dnap.nodesAtmSel)

        resname = ref_graph_data[wind_i][4]
        resid = ref_graph_data[wind_i][5]
        ref_str = f"resname {resname} and resid {resid} and segid ENZY"

        assert ref_str == sel_str

# @pytest.mark.parametrize(
#     ("num_cores", "ref_graph_data"), [
#         pytest.param(1,  [(217, 539), (217, 474)]),
#         pytest.param(2, [(217, 539, 0.023, 0.174),
#                              (217, 474, 0.0202, 0.1711)])
#     ])
# def test_calc_graph_parallel(dnap_omp, num_cores, ref_graph_data):
#
#     dnap = load_sys_solv_mode(dnap_omp, False, "all")
#
#     dnap.calcCartesian(backend="serial", verbose=0)
#
#     # calculate optimal paths
#     print("Calculating optimal paths...")
#     dnap.calcOptPaths(ncores=num_cores)
#
#     print("Calculating edge betweeness...")
#     # calculate betweeness values
#     dnap.calcBetween(ncores=num_cores)
#
