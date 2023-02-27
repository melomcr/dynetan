import pytest

from dynetan.contact import get_lin_index_numba


class TestContactClass:

    @pytest.mark.parametrize(("src", "trg", "dim", "res"), [
        (0, 5, 10, 4),
        (1, 5, 10, 12),
        pytest.param(1, 5, 10, 0, marks=pytest.mark.xfail)])
    def test_get_lin_index_numba(self, src, trg, dim, res):
        testRes = get_lin_index_numba(src, trg, dim)
        assert testRes == res
