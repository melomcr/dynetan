import pytest

from dynetan.proctraj import is_proteic


@pytest.mark.parametrize(("resname", "return_val"), [
    ("XXX", False),
    ("ALA", True), ("ARG", True), ("ASN", True), ("ASP", True), ("CYS", True),
    ("GLN", True), ("GLU", True), ("GLY", True), ("HSD", True), ("ILE", True),
    ("LEU", True), ("LYS", True), ("MET", True), ("PHE", True), ("PRO", True),
    ("SER", True), ("THR", True), ("TYR", True), ("VAL", True), ("TRP", True)
])
def test_is_proteic(dnap_omp, resname, return_val):
    """
    This will test the assumption that MDAnalysis will keep a consistent list
    of residue names that covers the standard 20 amino acids

    We take this list from the larger list in:
    https://docs.mdanalysis.org/stable/documentation_pages/core/selection.html
    https://userguide.mdanalysis.org/stable/standard_selections.html#proteins
    """
    assert is_proteic(resname) == return_val


def test_node_groups_new_res(dnap_omp):
    """This will test the OMP user-defined residue is being correctly loaded
        into node groups
    """
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    loaded_groups = dnap_omp.resNodeGroups

    # Check correctly loaded atom groups for OMP
    assert "N1" in loaded_groups['OMP'].keys()
    assert "P" in loaded_groups['OMP'].keys()

    atm_grp = {'C2', 'C4', 'C5', 'C6', 'C7', 'N1', 'N3', 'O2', 'O4', 'OA', 'OB'}
    assert loaded_groups['OMP']["N1"] == atm_grp

    atm_grp = {"O5'", "O2'", 'OP1', 'P', "C4'", "O3'", 'OP3', "C3'", 'OP2',
               "C5'", "O4'", "C1'", "C2'"}
    assert loaded_groups['OMP']["P"] == atm_grp


@pytest.mark.xfail(raises=Exception)
def test_node_groups_unk_res(dnap_omp):
    # Overwrite node group to force ERROR
    # "Residue OMP does not have a defined node group!"
    node_grps = {"TIP3": {}}
    node_grps["TIP3"]["OH2"] = set("OH2 H1 H2".split())
    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)


@pytest.mark.xfail(raises=Exception)
def test_node_groups_unk_node_atom(dnap_omp):
    # This function EXPANDS the OMP "P" atom group to raise an exception
    # where "Atoms were not assigned to any node!"

    # Overwrite node group to force ERROR
    node_grps = {"OMP": {}, "TIP3": {}}

    # Define nodes and atom groups for the ligand
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["PX"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

    node_grps["TIP3"]["OH2"] = set("OH2 H1 H2".split())
    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False, autocomp_groups=False)


@pytest.mark.xfail(raises=Exception)
def test_node_groups_unk_atoms(dnap_omp):
    # This function REDUCES the OMP "P" atom group to raise an exception
    # where "Found atoms not assigned to any node!"

    # Overwrite node group to force ERROR
    node_grps = {"OMP": {}, "TIP3": {}}

    # Define nodes and atom groups for the ligand
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4'".split())

    node_grps["TIP3"]["OH2"] = set("OH2 H1 H2".split())
    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False, autocomp_groups=False)


def test_node_groups_unk_atoms_verb(dnap_omp, capfd):
    # This function REDUCES the OMP "P" atom group to raise an exception
    # where "Found atoms not assigned to any node!"

    # We now check the verbosity output of the previous check.

    # Overwrite node group to force ERROR
    node_grps = {"OMP": {}, "TIP3": {}}

    # Define nodes and atom groups for the ligand
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4'".split())

    node_grps["TIP3"]["OH2"] = set("OH2 H1 H2".split())
    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    try:
        dnap_omp.prepareNetwork(verbose=2, autocomp_groups=False)
    except Exception:
        # We catch the exception to allow PyTest to verify the verbose output.
        pass

    captured = capfd.readouterr()

    # Main error message
    assert "ERROR: Atoms were not assigned to any node!" in captured.out

    # Atom indices for selected residues that were not assigned to any node group
    assert "[1633 1634 1643 1644 1645 1646 1647]" in captured.out

    test_str = "<Atom 1635: OT2 of type OC of resname LEU, resid 225 and segid ENZY>"
    assert test_str in captured.out

    test_str = "<Atom 1648: C1' of type CG3C51 of resname OMP, resid 301 and segid OMP>"
    assert test_str in captured.out


@pytest.mark.parametrize(("solv", "atm_grp"), [
        pytest.param(True,  {'OH2'}),
        pytest.param(False, {''}, marks=pytest.mark.xfail)])
def test_prepnet_water(dnap_omp, solv, atm_grp):
    """This will test if solvent residues are being correctly loaded into the
    atom groups.
    """

    # Overwrite node group to force ERROR
    node_grps = {"OMP": {}}

    # Define nodes and atom groups for the ligand
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())
    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=solv)

    dnap_omp.prepareNetwork(verbose=False, autocomp_groups=True)

    loaded_groups = dnap_omp.resNodeGroups

    assert 'TIP3' in loaded_groups.keys()

    # Check correctly loaded atom group for TIP3
    assert "OH2" in loaded_groups['TIP3'].keys()

    assert loaded_groups['TIP3']["OH2"] == atm_grp


def test_prepnet_groups(dnap_omp):
    """This will test the MDanalysis-defined protein residues are being correctly
    loaded into node groups.
    """

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False)

    dnap_omp.prepareNetwork(verbose=False)

    expected_keys = ['OMP', 'VAL', 'MET', 'ASN', 'ARG', 'LEU', 'ILE', 'ALA', 'ASP',
                     'THR', 'GLY', 'GLU', 'TYR', 'LYS', 'PRO', 'SER', 'PHE', 'CYS',
                     'HSD', 'GLN', 'TIP3']

    loaded_groups = dnap_omp.resNodeGroups

    for key in expected_keys:
        assert key in loaded_groups.keys()

    # Spot check amino acids
    # TYR
    assert "CA" in loaded_groups['TYR'].keys()

    atm_grp = {'C', 'CA', 'CB', 'CD1', 'CD2', 'CE1',
               'CE2', 'CG', 'CZ', 'N', 'O', 'OH'}
    assert loaded_groups['TYR']["CA"] == atm_grp

    # GLY
    assert "CA" in loaded_groups['GLY'].keys()

    atm_grp = {'CA', 'O', 'C', 'N'}
    assert loaded_groups['GLY']["CA"] == atm_grp

    # PRO
    assert "CA" in loaded_groups['PRO'].keys()

    atm_grp = {'CA', 'O', 'CB', 'CG', 'C', 'N', 'CD'}
    assert loaded_groups['PRO']["CA"] == atm_grp

    # PHE
    assert "CA" in loaded_groups['PHE'].keys()

    atm_grp = {'CA', 'CE1', 'O', 'CZ', 'CB', 'CG', 'CD1', 'CE2', 'C', 'CD2', 'N'}
    assert loaded_groups['PHE']["CA"] == atm_grp

    # GLN
    assert "CA" in loaded_groups['GLN'].keys()

    atm_grp = {'CA', 'O', 'CB', 'CG', 'NE2', 'C', 'N', 'OE1', 'CD'}
    assert loaded_groups['GLN']["CA"] == atm_grp


def test_prepnet_autocomplete(dnap_omp):
    """This will test the node group autocompletion feature is detecting atoms
    not explicitly hard-coded by users.
    """

    # Define nodes and atom group for a standard amino acid
    node_grps = {"ALA": {}, "OMP": {}}
    node_grps["ALA"]["CA"] = {'CA', 'O', 'C', 'N'}
    node_grps["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    node_grps["OMP"]["P"] = set(
        "P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

    dnap_omp.setNodeGroups(node_grps)

    dnap_omp.checkSystem()

    dnap_omp.selectSystem(withSolvent=False,
                          inputSelStr="protein or resname OMP or resname SOD")

    dnap_omp.prepareNetwork(verbose=False, autocomp_groups=True)

    loaded_groups = dnap_omp.resNodeGroups

    # Check the loaded group for ALA - should have hydrogen atoms
    assert "CA" in loaded_groups['ALA'].keys()

    atm_grp = {'HA', 'CA', 'HB2', 'HN', 'O', 'CB', 'HB3', 'C', 'HB1', 'N'}
    assert loaded_groups['ALA']["CA"] == atm_grp

    # Check the loaded group for GLY - should have hydrogen atoms
    assert "CA" in loaded_groups['GLY'].keys()

    atm_grp = {'CA', 'O', 'HN', 'HA2', 'C', 'N', 'HA1'}
    assert loaded_groups['GLY']["CA"] == atm_grp

    # Check the loaded group for OMP - should have hydrogen atoms
    assert "N1" in loaded_groups['OMP'].keys()
    atm_grp = {'C5', 'N1', 'C4', 'OB', 'C2', 'O2', 'O4', 'OA', 'H2', 'C6',
               'C7', 'H', 'N3'}
    assert loaded_groups['OMP']["N1"] == atm_grp

    # Check the loaded group for OMP - should have hydrogen atoms
    assert "P" in loaded_groups['OMP'].keys()
    atm_grp = {"O2'", "O3'", 'OP2', "C1'", "O5'", 'OP1', 'H6', 'H9', "C4'",
               'H10', "C3'", "C2'", "O4'", 'H12', 'H8', "C5'", 'H7', 'H4', 'P',
               'H5', 'OP3', 'H11', 'H3'}
    assert loaded_groups['OMP']["P"] == atm_grp

    # Check the loaded group for SOD - should have hydrogen atoms
    assert "SOD" in loaded_groups['SOD'].keys()

    atm_grp = {'SOD'}
    assert loaded_groups['SOD']["SOD"] == atm_grp


def test_prepnet_verb(dnap_omp, capfd):
    dnap_omp.checkSystem()
    dnap_omp.selectSystem(withSolvent=True, inputSelStr="protein", verbose=0)

    dnap_omp.prepareNetwork(verbose=1, autocomp_groups=True)

    captured = capfd.readouterr()

    assert "Selection string for atoms" in captured.out
    assert "protein" in captured.out
    assert "Preparing nodes..." in captured.out
    assert "Nodes are ready for network analysis." in captured.out


@pytest.mark.parametrize(("sel_str",
                          "NPshape",
                          "NPAuxshape"), [
        ("protein or resname OMP", (3327,), (217,)),
        ("protein", (3291,), (215,)),
        ("", (1928,), (486,))])
def test_prepnet_grp_indices(dnap_omp, sel_str, NPshape, NPAuxshape):
    dnap_omp.checkSystem()
    dnap_omp.selectSystem(withSolvent=True, inputSelStr=sel_str, verbose=0)
    dnap_omp.prepareNetwork(verbose=0, autocomp_groups=True)

    assert dnap_omp.nodeGroupIndicesNP.shape == NPshape
    assert dnap_omp.nodeGroupIndicesNPAux.shape == NPAuxshape
