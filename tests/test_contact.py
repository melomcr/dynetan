import pytest

from dynetan.contact import getLinIndexC
from dynetan.contact import getLinIndexNumba


class TestContactClass:

    @pytest.mark.parametrize(("src", "trg", "dim", "res"), [
        (0, 5, 10, 4),
        (1, 5, 10, 12),
        pytest.param(1, 5, 10, 0, marks=pytest.mark.xfail)])
    def test_get_lin_index_c(self, src, trg, dim, res):
        testRes = getLinIndexC(src, trg, dim)
        assert testRes == res

    @pytest.mark.parametrize(("src", "trg", "dim", "res"), [
        (0, 5, 10, 4),
        (1, 5, 10, 12),
        pytest.param(1, 5, 10, 0, marks=pytest.mark.xfail)])
    def test_get_lin_index_numba(self, src, trg, dim, res):
        testRes = getLinIndexNumba(src, trg, dim)
        assert testRes == res
