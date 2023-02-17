import pytest
import numpy as np
from dynetan.toolkit import getNodeFromSel
import dynetan.gencor as gc


@pytest.mark.xfail(raises=AssertionError)
def test_calc_cor_input(dnap_omp_loaded):
    dnap_omp_loaded.calcCor(ncores=-1)


@pytest.mark.parametrize("ncores", [1, 2, 10])
def test_calc_cor_force_calc(dnap_omp_loaded, ncores):
    dnap_omp_loaded.calcCor(ncores=ncores, verbose=0)

    # Find nodes for OMP
    nodes = getNodeFromSel("resname OMP",
                           dnap_omp_loaded.nodesAtmSel,
                           dnap_omp_loaded.atomToNode)

    # Select node in contact with OMP in both windows
    node_in_contact = 192

    # Save original correlations
    ref_corr = [0, 1]
    for i in range(2):
        ref_corr[i] = dnap_omp_loaded.corrMatAll[i, nodes[0], node_in_contact]

    # CHANGE CONNECTIVITY (Remove contact)
    for i in range(2):
        dnap_omp_loaded.contactMatAll[i, nodes[0], node_in_contact] = 0
        dnap_omp_loaded.contactMatAll[i, node_in_contact, nodes[0]] = 0

    # CHANGE CONNECTIVITY (ADD OMP P-N1 CONTACT)
    for i in range(2):
        dnap_omp_loaded.contactMatAll[i, nodes[0], nodes[1]] = 1
        dnap_omp_loaded.contactMatAll[i, nodes[1], nodes[0]] = 1

    # RECALCULATE CORRELATION WITH "FORCE_CALC" OFF.
    dnap_omp_loaded.calcCor(ncores=ncores, forceCalc=False, verbose=0)

    # Check that the same correlation exists between nodes no longer in contact
    # And that the OMP nodes are still are now correlated.
    for i in range(2):
        # No longer connected
        assert ref_corr[i] == dnap_omp_loaded.corrMatAll[i, nodes[0], node_in_contact]

        # New connection
        # (due to low sampling, window 1 has no correlation between OMP nodes)
        assert 0.2500220177999451 == dnap_omp_loaded.corrMatAll[0, nodes[0], nodes[1]]
        assert 0.0 == dnap_omp_loaded.corrMatAll[1, nodes[0], nodes[1]]

    # RECALCULATE CORRELATION WITH "FORCE_CALC" ON !!!
    dnap_omp_loaded.calcCor(ncores=ncores, forceCalc=True, verbose=2)

    # ASSERT NO CORRELATION BETWEEN OMP and node_in_contact
    for i in range(2):
        # No longer connected
        assert 0 == dnap_omp_loaded.corrMatAll[i, nodes[0], node_in_contact]


def test_calc_cor_par(dnap_omp_loaded):
    """
    This function mimics the beginning of `calcCor` to test the parallel methods
    used in parallel correlation calculation.
    """
    import queue
    import multiprocessing as mp

    # For 3D atom position data
    num_dims = 3
    n_frames = dnap_omp_loaded.workU.trajectory.n_frames
    n_winds  = dnap_omp_loaded.numWinds
    n_nodes  = dnap_omp_loaded.numNodes

    # Find nodes for OMP
    nodes = getNodeFromSel("resname OMP",
                           dnap_omp_loaded.nodesAtmSel,
                           dnap_omp_loaded.atomToNode)

    # CHANGE CONNECTIVITY (ADD OMP P-N1 CONTACT)
    for i in range(2):
        dnap_omp_loaded.contactMatAll[i, nodes[0], nodes[1]] = 1
        dnap_omp_loaded.contactMatAll[i, nodes[1], nodes[0]] = 1

    win_indx = 0

    win_len = int(np.floor(n_frames/n_winds))
    beg = int(win_indx * win_len)
    end = int((win_indx + 1) * win_len)

    # Initialize the correlation matrix with zeros
    dnap_omp_loaded.corrMatAll = np.zeros([n_winds, n_nodes, n_nodes],
                                          dtype=np.float64)

    traj: np.ndarray = np.ndarray([n_nodes, num_dims, win_len], dtype=np.float64)
    traj.fill(0)

    psi, phi = dnap_omp_loaded._prep_phi_psi(win_len)

    pair_array = dnap_omp_loaded._create_pair_list(win_indx=win_indx)

    gc.prep_mi_c(dnap_omp_loaded.workU, traj, beg, end, n_nodes, num_dims)

    # Create queues that feed processes with node pairs, and gather results.
    data_queue: queue.Queue = mp.Queue()
    results_queue: queue.Queue = mp.Queue()

    # Loads the node pairs in the input queue
    for atmList in pair_array:
        data_queue.put(atmList)
    data_queue.put([])

    gc.calc_cor_proc(traj,
                     win_len,
                     psi,
                     phi,
                     num_dims,
                     dnap_omp_loaded.kNeighb,
                     data_queue, results_queue)

    # Gathers all results.
    for _ in range(len(pair_array)):
        result = results_queue.get()

        node1 = result[0][0]
        node2 = result[0][1]
        corr = gc.mir_to_corr(result[1])

        dnap_omp_loaded.corrMatAll[win_indx, node1, node2] = corr
        dnap_omp_loaded.corrMatAll[win_indx, node2, node1] = corr

    # We only calculate correlations for window 0 in this test
    assert 0.2500220177999451 == dnap_omp_loaded.corrMatAll[0, nodes[0], nodes[1]]


@pytest.mark.parametrize(("ncores", "force", "counts"), [
    (1, False, [338, 384]),
    (10, False, [338, 384]),
    (1, True, [854, 847]),
    (10, True, [854, 847])])
def test_calc_cor_verb(dnap_omp_loaded, capsys, ncores, force, counts):
    # First run is default
    dnap_omp_loaded.calcCor(ncores=ncores, forceCalc=False, verbose=0)

    # Second run uses the arguments and has verbosity output
    dnap_omp_loaded.calcCor(ncores=ncores, forceCalc=force, verbose=2)

    # Check verbosity output
    captured = capsys.readouterr()

    assert "Calculating correlations" in captured.out
    assert "Using window length of " in captured.out

    if ncores == 1:
        assert "- > Using single-core implementation." in captured.out
    else:
        test_str = f"- > Using multi-core implementation with {ncores} threads."
        assert test_str in captured.out

    if not force:
        test_str = "Removing 516 pairs with pre-calculated correlations in window 0."
        assert test_str in captured.out

        test_str = "Removing 463 pairs with pre-calculated correlations in window 1."
        assert test_str in captured.out

    # Output from second calcCorr run
    test_str = f"{counts[0]} new correlations to be calculated in window 0."
    assert test_str in captured.out

    test_str = f"{counts[1]} new correlations to be calculated in window 1."
    assert test_str in captured.out


@pytest.mark.parametrize(("ncores", "force"), [
    (1, False), (10, False)])
def test_calc_cor_verb_2(dnap_omp_loaded, capsys, ncores, force):
    """
    This function will test a particular case where correlation calculation is
    asked twice without "force=True" AND there is no pair of nodes in contact
    with correlation equal to zero.

    This will prompt a message (given verbosity > 1) and a branch where no new
    correlation calculation will be executed.

    Alternatively, if there was a pait of nodes in contact that had previously
    been evaluated with zero correlation, the correlation would be evaluated
    again.

    This allows for NEW contacts to be introduced in the contact matrix and their
    correlations can be evaluated in sequential rounds of calcCorr.
    """
    # First run is default
    dnap_omp_loaded.calcCor(ncores=1, forceCalc=False, verbose=0)

    for win in range(2):
        contacts = dnap_omp_loaded.contactMatAll[win, :, :].copy()
        contacts = np.triu(contacts)
        contactsL = np.where(contacts > 0)
        contactsL = np.asarray(contactsL).T

        # Create a list of node pairs for which we have a contact but no correlation
        no_corr_contacts = [(n1, n2) for n1, n2 in contactsL if
                            dnap_omp_loaded.corrMatAll[win, n1, n2] == 0]

        # Artificially introduce minimal correlation between such nodes.
        for n1, n2 in no_corr_contacts:
            dnap_omp_loaded.corrMatAll[win, n1, n2] = 0.001
            dnap_omp_loaded.corrMatAll[win, n2, n1] = 0.001

    # Second run uses the arguments and has verbosity output
    dnap_omp_loaded.calcCor(ncores=ncores, forceCalc=force, verbose=2)

    # Capture verbosity output
    captured = capsys.readouterr()

    assert "No new correlations to be calculated" in captured.out


@pytest.mark.xfail(raises=Exception)
def test_calc_cor_sanity_check(dnap_omp_loaded):
    dnap_omp_loaded.calcCor(ncores=1)

    dnap_omp_loaded.corrMatAll[1, 0, 1] = 1
    dnap_omp_loaded.corrMatAll[1, 1, 0] = 0

    dnap_omp_loaded._corr_mat_symmetric()
