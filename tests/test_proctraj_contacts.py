import pytest
import numpy as np


@pytest.mark.parametrize(("select_str", "in_memory", "position"), [
    pytest.param("", True,
                 np.array([-19.168497, -6.3753138, -3.1209965],
                          dtype=np.float32)),
    pytest.param("protein", True,
                 np.array([-19.168983, -6.376695, -3.1217244],
                          dtype=np.float32)),
    pytest.param("resid 61 and resname LYS", True,
                 np.array([-19.030607, -6.5320506, -3.1086228],
                          dtype=np.float32))
])
def test_align(dnap_omp, select_str, in_memory, position):
    """This will test the alignment selection, and spot check the final position
    of the last atom in an arbitrarily selected residue to verify that the
    alignment was carried out consistently.
    """
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.alignTraj(selectStr=select_str, inMemory=in_memory)

    _ = dnap_omp.getU().trajectory[-1]
    test_atm_sel = dnap_omp.getU().select_atoms("resid 61 and resname LYS")
    test_position = test_atm_sel.positions[-1, :]

    assert np.allclose(position, test_position)


def test_align_verb(dnap_omp, capfd):
    """This will test the alignment verbose output.
    """
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.alignTraj(selectStr="protein and resid 61", verbose=2)

    captured = capfd.readouterr()

    assert "Using alignment selection string:" in captured.out
    assert "protein and resid 61" in captured.out


@pytest.mark.parametrize(("solv", "stride", "num_contacts"), [
    pytest.param(True,  1, 1898),
    pytest.param(True,  2, 1958),
    pytest.param(False, 1, 1075)])
def test_find_contacts(dnap_omp, solv, stride, num_contacts):
    """This will test the contact detection main function.

    The default contact persistence is 0.75.
    There are 20 frames in this trajectory.
    The value used for numWinds is 2.
    Therefore, each window is 10 frames long.

    contactCutoff = (winLen / stride) * contactPersistence

    """
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=solv)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.alignTraj()

    dnap_omp.findContacts(stride=stride)

    contacts = len(np.asarray(np.where(np.triu(dnap_omp.contactMat) > 0)).T)

    assert contacts == num_contacts


@pytest.mark.xfail(raises=Exception)
def test_check_contact_mat(dnap_omp):
    """
    Tests sanity checks and validity of contact matrix
    """
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.alignTraj()

    dnap_omp.findContacts(stride=1, verbose=0)

    dnap_omp.contactMat[0, 1] = 1
    dnap_omp.contactMat[1, 0] = 0

    dnap_omp.checkContactMat()


def test_find_contacts_verb(dnap_omp, capfd):
    """
    Tests detection of isolated nodes and fraction of connected nodes.
    We force the package to select ionic residues but do not select for
    solvent residues, leaving ions isolated.

    """
    dnap_omp.setSegIDs(["OMP", "ENZY", "ION"])

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.alignTraj()

    dnap_omp.findContacts(stride=2, verbose=2)

    captured = capfd.readouterr()

    # Output from "findContacts"
    assert "Starting contact calculation for window" in captured.out
    assert "Time for contact calculation" in captured.out

    # Output from "_contactTraj"
    assert "Using distance calculation mode" in captured.out
    assert "Allocating temporary distance" in captured.out
    assert "Allocated temporary distance " in captured.out
    assert "Allocated temporary NODE distance " in captured.out

    assert "Checking frames 0 to 10 with stride 2." in captured.out
    assert "Checking frames 10 to 20 with stride 2." in captured.out

    assert "Calculating contacts for timestep " in captured.out

    # Output from "checkContactMat"
    assert "We found 7 nodes with no contacts." in captured.out
    assert "We found 1078 contacting pairs" in captured.out
    assert "out of 25200 total pairs of nodes." in captured.out


@pytest.mark.parametrize(("dist_mode", "mode_str", "fill_str"), [
    pytest.param("all",
                 "Using distance calculation mode: all",
                 "Filling array with zeros."),
    pytest.param("capped",
                 "Using distance calculation mode: capped",
                 "Filling array twice the cutoff distance")
])
def test_find_contacts_verb_dist_mode(dnap_omp, capfd,
                                      dist_mode, mode_str, fill_str):
    dnap_omp.setDistanceMode(dist_mode)

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.findContacts(stride=2, verbose=2)

    captured = capfd.readouterr()

    assert mode_str in captured.out
    assert fill_str in captured.out


@pytest.mark.parametrize(("end_frame", "output_str"), [
    pytest.param(-1, "Checking frames 0 to 20 with stride 2."),
    pytest.param(10, "Checking frames 0 to 10 with stride 2.")
])
def test_find_contacts_verb_frame_sel(dnap_omp, capfd, end_frame, output_str):
    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.findContacts(stride=2, verbose=2)

    dnap_omp._contactTraj(dnap_omp.contactMatAll[0, :, :],
                          beg=0, end=end_frame, stride=2, verbose=2)

    captured = capfd.readouterr()

    assert output_str in captured.out


@pytest.mark.parametrize(("not_same_res",
                          "not_consec_res",
                          "remove_isolated",
                          "init_nodes", "final_nodes",
                          "contact_pairs", "total_pairs"), [
    pytest.param(True, True, True,
                 225, 218,
                 863, 23653),  # Removes contacts between consecutive amino acids
    pytest.param(True, False, True,
                 225, 218,
                 1077, 23653),  # Default parameters
    pytest.param(False, False, True,
                 225, 218,
                 1078, 23653),  # Allows OMP nodes to connect
    pytest.param(False, False, False,
                 225, 225,
                 1078, 25200),  # Keeps nodes with no contacts
])
def test_filter_contacts(dnap_omp, capfd,
                         not_same_res, not_consec_res, remove_isolated,
                         init_nodes, final_nodes,
                         contact_pairs, total_pairs):
    dnap_omp.setSegIDs(["OMP", "ENZY", "ION"])

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.findContacts(stride=2, verbose=0)

    assert dnap_omp.numNodes == init_nodes

    dnap_omp.filterContacts(notSameRes=not_same_res,
                            notConsecutiveRes=not_consec_res,
                            removeIsolatedNodes=remove_isolated)

    assert dnap_omp.numNodes == final_nodes

    dnap_omp.checkContactMat()

    captured = capfd.readouterr()

    test_str = f"{contact_pairs} contacting pairs out of {total_pairs}"
    assert test_str in captured.out


def test_filter_contacts_verb(dnap_omp, capfd):
    dnap_omp.setSegIDs(["OMP", "ENZY", "ION"])

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    dnap_omp.findContacts(stride=2, verbose=0)

    dnap_omp.filterContacts(notSameRes=True,
                            notConsecutiveRes=True,
                            removeIsolatedNodes=True,
                            verbose=2)

    captured = capfd.readouterr()

    print(" END ------ ")

    assert "Window:" in captured.out
    assert "We found 854 contacting pairs" in captured.out

    assert "Removing isolated nodes" in captured.out

    assert "We found 7 nodes with no contacts." in captured.out
    test_str = "Atom 1666: SOD of type SOD of resname SOD, resid 7 and segid ION"
    assert test_str in captured.out

    assert "Isolated nodes removed." in captured.out

    assert "Updating Universe to reflect new node selection" in captured.out

    sel_str = "(resname TIP3 and name OH2)"
    assert sel_str in captured.out

    sel_str = "(resname OMP and name N1 P)"
    assert sel_str in captured.out

    sel_str = "(resname ALA and name CA)"
    assert sel_str in captured.out

    sel_str = "(resname SOD and name SOD)"
    assert sel_str in captured.out
