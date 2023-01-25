from MDAnalysis import lib
from dynetan.toolkit import getLinIndexC


class TestToolkitClass:

    def test_diag_func(self):
        from dynetan.toolkit import diagnostic

        assert diagnostic() == lib.distances.USED_OPENMP

    def test_get_lin_index_c(self):

        testsList = [
            (0, 5, 10, 4),
            (1, 5, 10, 12)
        ]

        for src, trg, dim, res in testsList:
            testRes = getLinIndexC(src, trg, dim)

            assert testRes == res
