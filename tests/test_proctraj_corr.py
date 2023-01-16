import pytest
from dynetan.toolkit import getNodeFromSel
import numpy as np

from .test_proctraj_checksys_selectsys import test_data_dir  # NOQA - PyCharm
from .test_proctraj_checksys_selectsys import dnap_omp  # NOQA - PyCharm


@pytest.fixture
def dnap_omp_loaded(dnap_omp):
    dnap_omp.checkSystem()

    dnap_omp.selectSystem(with_solvent=False)

    dnap_omp.prepareNetwork(verbose=0)

    dnap_omp.alignTraj()

    dnap_omp.findContacts(stride=2, verbose=0)

    dnap_omp.filterContacts(notSameRes=True,
                            notConsecutiveRes=True,
                            removeIsolatedNodes=True,
                            verbose=0)

    return dnap_omp


@pytest.mark.xfail(raises=AssertionError)
def test_calc_cor_input(dnap_omp_loaded):
    dnap_omp_loaded.calcCor(ncores=-1)


def test_calc_cor_force_calc(dnap_omp_loaded):
    dnap_omp_loaded.calcCor(ncores=1)

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
    dnap_omp_loaded.calcCor(ncores=1, forceCalc=False)

    # Check that the same correlation exists between nodes no longer in contact
    # And that the OMP nodes are still are now correlated.
    for i in range(2):
        # No longer connected
        assert ref_corr[i] == dnap_omp_loaded.corrMatAll[i, nodes[0], node_in_contact]

        # New connection
        # (due to low sampling, window 1 has no correlation between OMP nodes)
        assert 0.3918299377643048 == dnap_omp_loaded.corrMatAll[0, nodes[0], nodes[1]]
        assert 0.0 == dnap_omp_loaded.corrMatAll[1, nodes[0], nodes[1]]

    # RECALCULATE CORRELATION WITH "FORCE_CALC" ON !!!
    dnap_omp_loaded.calcCor(ncores=1, forceCalc=True)

    # ASSERT NO CORRELATION BETWEEN OMP and node_in_contact
    for i in range(2):
        # No longer connected
        assert 0 == dnap_omp_loaded.corrMatAll[i, nodes[0], node_in_contact]


@pytest.mark.xfail(raises=Exception)
def test_calc_cor_sanity_check(dnap_omp_loaded):
    dnap_omp_loaded.calcCor(ncores=1)

    dnap_omp_loaded.corrMatAll[1, 0, 1] = 1
    dnap_omp_loaded.corrMatAll[1, 1, 0] = 0

    dnap_omp_loaded._corr_mat_symmetric()
