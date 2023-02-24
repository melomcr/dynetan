import pytest

from dynetan.toolkit import getCartDist


dist_list = [
    [0,   0,   [0.0   ,  0.0   , 0.0   ,  0.0   ]],  # noqa: E202, E203
    [0,   1,   [1.3497,  0.0038, 1.3443,  1.3606]],
    [0,   10,  [18.3727, 0.1023, 18.2059, 18.6711]],
    [215, 192, [2.8618,  0.0233, 2.8145,  2.9069]],
    [216, 192, [6.8436,  0.2412, 6.3493,  7.4415]]
]
dist_list_capped = [
    [0,   1,   [1.3497, 0.0038, 1.3443,  1.3606]],
    [0,   10,  [9.0,    0.0,    9.0,     9.0]],
    [215, 192, [2.8618, 0.0233, 2.8145,  2.9069]],
    [216, 192, [9.0,    0.0,    9.0,     9.0]]
]
solv_dist_list = [
    [217, 158, [3.2461, 0.3134, 2.6309, 3.9079]],
    [484, 67,  [2.7779, 0.1157, 2.5903, 3.0874]]
]


def load_sys_solv_mode(dnap_base, solv, mode):
    # Extra setting to test contacts and distance calculation
    dnap_base.setDistanceMode(mode)

    dnap_base.checkSystem()

    dnap_base.selectSystem(withSolvent=solv)

    dnap_base.prepareNetwork(verbose=0)

    dnap_base.alignTraj()

    dnap_base.findContacts(stride=2, verbose=0)

    dnap_base.filterContacts(notSameRes=True,
                             notConsecutiveRes=True,
                             removeIsolatedNodes=True,
                             verbose=0)

    dnap_base.calcCor(ncores=1)

    return dnap_base


@pytest.mark.parametrize(
    ("solv", "backend", "mode", "node_dists_shape", "distances"), [
        pytest.param(True,  "serial", "all", (4, 117855),
                     dist_list + solv_dist_list),
        pytest.param(False, "serial", "all", (4, 23436),
                     dist_list),
        pytest.param(True,  "openmp", "all", (4, 117855),
                     dist_list + solv_dist_list),
        pytest.param(False, "openmp", "all", (4, 23436),
                     dist_list),
        pytest.param(True,  "serial", "capped", (4, 117855),
                     dist_list_capped + solv_dist_list),
        pytest.param(False, "serial", "capped", (4, 23436),
                     dist_list_capped)])
def test_calc_cartesian(dnap_omp,
                        solv, backend, mode,
                        node_dists_shape, distances):

    dnap = load_sys_solv_mode(dnap_omp, solv, mode)

    dnap.calcCartesian(backend=backend, verbose=0)

    assert dnap.nodeDists.shape == node_dists_shape

    for dist in distances:
        src, trgt, ref_vec = dist

        for dist_type in range(4):
            cart_dist = getCartDist(src, trgt,
                                    dnap.numNodes,
                                    dnap.nodeDists,
                                    dist_type)
            cart_dist = round(cart_dist, 4)

            assert ref_vec[dist_type] == cart_dist


@pytest.mark.parametrize(
    ("solv", "mode", "nodes_atms"), [
        pytest.param(True,  "all",    (486, 1928)),
        pytest.param(False, "all",    (217, 1659)),
        pytest.param(True,  "capped", (486, 1928, 14771)),
        pytest.param(False, "capped", (217, 1659, 13765))])
def test_calc_cartesian_verb(dnap_omp, capfd, solv, mode, nodes_atms):

    # Here we only test verbosity output, which is the same with either backend
    backend = "serial"

    num_nodes = nodes_atms[0]
    num_atoms = nodes_atms[1]

    dnap = load_sys_solv_mode(dnap_omp, solv, mode)

    dnap.calcCartesian(backend=backend, verbose=2)

    # Check verbosity output for multicore run
    captured = capfd.readouterr()

    test_str = "Calculating cartesian distances"
    assert test_str in captured.out

    test_str = "Sampling a total of 4 frames from 2 windows (2 per window)"
    assert test_str in captured.out

    # Form contact.calcDistances method ###########

    test_str = f"There are {num_nodes} nodes and {num_atoms} atoms in this system."
    assert test_str in captured.out

    num_elements = int(num_atoms * (num_atoms - 1) / 2)

    test_str = f"creating array with {num_elements} elements"
    assert test_str in captured.out

    test_str = "Time for matrix"
    assert test_str in captured.out

    if mode == "all":
        test_str = "running self_distance_array"
        assert test_str in captured.out

    elif mode == "capped":
        test_str = "running self_capped_distance"
        assert test_str in captured.out

        num_pairs = nodes_atms[2]

        # Sampled Frame 1
        test_str = f"Found {num_pairs} pairs and {num_pairs} distances"
        assert test_str in captured.out

        test_str = "loading distances in array"
        assert test_str in captured.out

        test_str = "Loaded 13000 distances"
        assert test_str in captured.out

        test_str = "Time for 13000 distances:"
        assert test_str in captured.out

        test_str = "Time for loading distances:"
        assert test_str in captured.out

    test_str = "Time for contact calculation:"
    assert test_str in captured.out

    test_str = "running atm_to_node_dist"
    assert test_str in captured.out

    test_str = "Time for atm_to_node_dist:"
    assert test_str in captured.out
