import pytest
import networkx as nx

from .test_proctraj_cartesian import load_sys_solv_mode


class TestGraph:
    ligandSegID = "OMP"

    def test_calc_graph(self, dnap_omp):
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

    @pytest.mark.parametrize("window", [
        pytest.param(0),
        pytest.param(1),
        pytest.param(-1, marks=pytest.mark.xfail),
        pytest.param(10, marks=pytest.mark.xfail)])
    def test_get_degree_dict(self, dnap_omp, window):
        """
        Tests getDegreeDict for window out of range
        """
        dnap = load_sys_solv_mode(dnap_omp, False, "all")

        dnap.calcGraphInfo()

        dnap.getDegreeDict(window)

    def test_calc_opt_path_par_func(self, dnap_omp):
        """
        This test will mimic the `calcOptPaths` method and test the function used
        to calculate optimal paths in parallel.
        """
        import copy
        import queue
        import numpy as np
        import multiprocessing as mp
        from dynetan import network as nw
        from dynetan.toolkit import getNodeFromSel, getPath

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        dnap.distsAll = np.zeros([dnap.numWinds, dnap.numNodes, dnap.numNodes],
                                 dtype=np.float64)
        dnap.preds = {}
        for i in range(dnap.numWinds):
            dnap.preds[i] = 0

        inQueue: queue.Queue = mp.Queue()
        outQueue: queue.Queue = mp.Queue()

        for win in range(dnap.numWinds):
            inQueue.put(win)
        inQueue.put(-1)

        nw.calcOptPathPar(dnap.nxGraphs, inQueue, outQueue)

        for _ in range(dnap.numWinds):
            result = outQueue.get()
            dnap.distsAll[result[0], :, :] = np.copy(result[1])
            dnap.preds[result[0]] = copy.deepcopy(result[2])

        trgtNodes = getNodeFromSel("segid " + self.ligandSegID,
                                   dnap.nodesAtmSel,
                                   dnap.atomToNode)

        enzNode = getNodeFromSel("segid ENZY and resname GLU and resid 115",
                                 dnap.nodesAtmSel,
                                 dnap.atomToNode)

        # For window 0
        ref_list = [104, 74, 76, 58, 85, 216]

        # First using the toolkit function
        opt_path = getPath(trgtNodes[1], enzNode[0],
                           dnap.nodesAtmSel, dnap.preds, 0)

        assert list(opt_path) == ref_list

    @pytest.mark.parametrize(
        "num_cores", [
            pytest.param(1, ),
            pytest.param(2, ),
            pytest.param(-1, marks=pytest.mark.xfail),
        ])
    def test_calc_paths_parallel(self, dnap_omp, num_cores):

        from dynetan.toolkit import getNodeFromSel, getPath

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        # calculate optimal paths
        dnap.calcOptPaths(ncores=num_cores)

        trgtNodes = getNodeFromSel("segid " + self.ligandSegID,
                                   dnap.nodesAtmSel,
                                   dnap.atomToNode)

        enzNode = getNodeFromSel("segid ENZY and resname GLU and resid 115",
                                 dnap.nodesAtmSel,
                                 dnap.atomToNode)

        # For window 0
        ref_list = [104, 74, 76, 58, 85, 216]

        # First using the toolkit function
        opt_path = getPath(trgtNodes[1], enzNode[0],
                           dnap.nodesAtmSel, dnap.preds, 0)

        assert list(opt_path) == ref_list

        # Then using the object function
        opt_path = dnap.getPath(enzNode[0], trgtNodes[1], 0)

        assert list(opt_path) == ref_list

        # For window 1
        ref_list = [104, 100, 98, 62, 84, 110, 112, 216]

        # First using the toolkit function
        opt_path = getPath(trgtNodes[1], enzNode[0],
                           dnap.nodesAtmSel, dnap.preds, 1)

        assert list(opt_path) == ref_list

        # Then using the object function
        opt_path = dnap.getPath(enzNode[0], trgtNodes[1], 1)

        assert list(opt_path) == ref_list

        # Extra check for the toolkit version
        # Check identical source and target nodes
        opt_path = getPath(0, 0, dnap.nodesAtmSel, dnap.preds, 0)
        assert len(opt_path) == 0

        # Check special case where there is no path connecting a specific node
        opt_path = getPath(92, 1, dnap.nodesAtmSel, dnap.preds, 0)
        assert len(opt_path) == 0

        # Check special case where there is no path between two nodes
        # We use the isolated node as target
        opt_path = getPath(104, 92, dnap.nodesAtmSel, dnap.preds, 0)
        assert len(opt_path) == 0

        # Check case where two nodes are from same residue
        opt_path = getPath(trgtNodes[0], trgtNodes[1],
                           dnap.nodesAtmSel, dnap.preds, 0)
        assert len(opt_path) == 0

    def test_calc_betweens_par_func(self, dnap_omp):
        """
        This test will mimic the `calcBetween` method and test the function used
        to calculate betweenness in parallel.
        """
        from itertools import islice
        import copy
        import queue
        import multiprocessing as mp
        from dynetan import network as nw

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        dnap.btws = {}

        inQueue: queue.Queue = mp.Queue()
        outQueue: queue.Queue = mp.Queue()

        for win in range(dnap.numWinds):
            inQueue.put(win)
        inQueue.put(-1)

        nw.calcBetweenPar(dnap.nxGraphs, inQueue, outQueue)

        for _ in range(dnap.numWinds):
            result = outQueue.get()
            dnap.btws[result[0]] = copy.deepcopy(result[1])

        # For window 0
        # (32, 57) 0.05792 0.13400926315561176
        for node_pair, betweenness in islice(dnap.btws[0].items(), 1):
            node_i = node_pair[0]
            node_j = node_pair[1]
            btw = round(betweenness, 5)

            assert node_i == 32
            assert node_j == 57
            assert btw == 0.05792

    @pytest.mark.parametrize(
        "num_cores", [
            pytest.param(1, ),
            pytest.param(2, ),
            pytest.param(-1, marks=pytest.mark.xfail),
        ])
    def test_calc_betweens_parallel(self, dnap_omp, num_cores):
        from itertools import islice

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        # Calculate betweeness values
        dnap.calcBetween(ncores=num_cores)

        # For window 0
        # (32, 57) 0.05792 0.13400926315561176
        for node_pair, betweenness in islice(dnap.btws[0].items(), 1):
            node_i = node_pair[0]
            node_j = node_pair[1]
            btw = round(betweenness, 5)

            assert node_i == 32
            assert node_j == 57
            assert btw == 0.05792

        # For window 1
        # (84, 110) 0.07405 0.13400926315561176
        for node_pair, betweenness in islice(dnap.btws[1].items(), 1):
            node_i = node_pair[0]
            node_j = node_pair[1]
            btw = round(betweenness, 5)

            assert node_i == 84
            assert node_j == 110
            assert btw == 0.07405

    def test_calc_eigen(self, dnap_omp):

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        ##############################

        dnap.calcEigenCentral()

        # Window, Node, Eigenvector Centrality
        ref_vals = [[0, 0,   0.2518],
                    [0, 215, 0.10615],
                    [0, 216, 0.04137],
                    [1, 0,   0.00585],
                    [1, 215, 0.00055],
                    [1, 216, 0.00731]
                    ]

        for window, node, ref_val in ref_vals:
            ev_val = round(dnap.nxGraphs[window].nodes[node]['eigenvector'], 5)
            assert ev_val == ref_val

    @pytest.mark.parametrize(
        "with_eigen", [
            pytest.param(True, ),
            pytest.param(False, ),
        ])
    def test_calc_communities(self, dnap_omp, with_eigen):
        """
        Community calculation uses a stochastic algorithm, so the number of
        communities and their labels can vary between executions.

        We will test if communities were assigned, and (if they were calculated
        after eigenvector centralities were calculated) if the first community
        was

        """
        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()

        # dnap.calcOptPaths(ncores=2)
        # dnap.calcBetween(ncores=2)

        ##############################

        if with_eigen:
            dnap.calcEigenCentral()

        dnap.calcCommunities()

        ##############################

        # Check that communities were assigned to nodes
        for window in range(2):
            for node in dnap.nxGraphs[window].nodes:
                assert 'modularity' in dnap.nxGraphs[window].nodes[node].keys()

            # We expect multiple community labels
            assert len(dnap.nodesComm[window]["commLabels"])

            assert "commOrderSize" in dnap.nodesComm[window].keys()

            if with_eigen:
                assert "commOrderEigenCentr" in dnap.nodesComm[window].keys()

                # Here we can test if the first community actually has the node
                # with the highest eigenvalue centrality
                ev_tmp = [(node, dnap.nxGraphs[window].nodes[node]['eigenvector'])
                          for node in dnap.nxGraphs[window].nodes]
                ev_tmp.sort(key=lambda x: x[1], reverse=True)
                # print(f"Node {ev_tmp[0][0]} centrality {round(ev_tmp[0][1], 5)}")
                ref_ev = round(ev_tmp[0][1], 5)

                first_com = dnap.nodesComm[window]["commOrderEigenCentr"][0]
                first_node = dnap.nodesComm[window]["commNodes"][first_com][0]

                val = round(dnap.nxGraphs[window].nodes[first_node]['eigenvector'], 5)
                assert val == ref_ev
