import pytest

from dynetan.viz import getCommunityColors
from dynetan.viz import prepTclViz

import nglview as nv
import os
import pandas as pd
from io import StringIO

from .test_proctraj_cartesian import load_sys_solv_mode


@pytest.fixture(scope="session")
def tmp_dna_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("tcl_script_folder")


class TestVizMethods:
    ligandSegID = "OMP"

    rawcolordata = u"""\
        ID    R    G    B
    0   40  128    0    0
    1   41   32  178  170
    2   42  139    0  139
    3   43  245  222  179
    4   44  220   20   60
    5   45  138   43  226
    6   46  245  255  250
    7   47   70  130  180
    8   48  219  112  147
    9   49  255  127   80
    10  50  255  182  193
    11  51  210  105   30
    12  52    0  255  255
    13  53  221  160  221
    14  54  205   92   92
    15  55  216  191  216
    16  56    0  139  139
    17  57  255  140    0
    18  58  184  134   11
    19  59  255  218  185
    20  60  178   34   34
    21  61    0    0    0
    22  62  240  255  240
    23  63  245  245  245
    24  64   30  144  255
    25  65  240  128  128
    26  66    0  250  154
    27  67   75    0  130
    28  68  112  128  144
    29  69  255   69    0
    30  70  255  192  203
    31  71  255  215    0
    32  72  100  149  237
    33  73  128  128    0
    34  74  238  232  170
    35  75  135  206  250
    36  76  189  183  107
    37  77  255  255    0
    38  78   72  209  204
    39  79  154  205   50
    40  80   25   25  112
    41  81    0  100    0
    42  82  250  235  215
    43  83  255   20  147
    44  84  173  255   47
    45  85  224  255  255
    46  86  152  251  152
    47  87  147  112  219
    48  88  139   69   19
    49  89  220  220  220
    """

    def test_get_community_colors(self):
        commColors = getCommunityColors()

        rawdataDF = pd.read_csv(StringIO(self.rawcolordata),
                                index_col=0,
                                sep=r"\s+",
                                engine='python')

        assert commColors.equals(rawdataDF)

    def test_prep_tcl_viz(self, tmp_dna_test_dir):
        """
            Test the
        """

        base_name = "networkData"
        num_steps = 12345
        ligand_segid = "ABCD"

        prepTclViz(base_name=base_name,
                   num_winds=num_steps,
                   ligand_segid=ligand_segid,
                   trg_dir=tmp_dna_test_dir)

        new_file_path = os.path.join(tmp_dna_test_dir, "network_view_2.tcl")

        assert os.path.isfile(new_file_path)

        test_lines = ["set basename {}".format(base_name),
                      "set numstep  {}".format(num_steps),
                      "set ligand   {}".format(ligand_segid)]

        with open(new_file_path, "r") as in_test_file:
            lines = in_test_file.readlines()
            lines = [line.strip() for line in lines]

        for test_line in test_lines:
            assert test_line in lines

    def test_view_path(self, dnap_omp):
        """
            This will test the NGLview function that creates representations.
            We cannot test the visual output, but can test argument
            compatibility anf function operation.
        """
        from dynetan.toolkit import getNodeFromSel
        from dynetan.toolkit import getPath
        from dynetan.viz import viewPath

        dnap = load_sys_solv_mode(dnap_omp, False, "all")
        dnap.calcGraphInfo()
        dnap.calcOptPaths(ncores=1)
        dnap.calcBetween(ncores=1)

        enzNode = getNodeFromSel("segid ENZY and resname GLU and resid 115",
                                 dnap.nodesAtmSel,
                                 dnap.atomToNode)
        trgtNodes = getNodeFromSel("segid " + self.ligandSegID,
                                   dnap.nodesAtmSel,
                                   dnap.atomToNode)

        optpath = getPath(trgtNodes[1], enzNode[0], dnap.nodesAtmSel, dnap.preds)

        w = nv.show_mdanalysis(dnap.getU().select_atoms("all"))

        # Test creation of a path between two different nodes
        viewPath(w, optpath, dnap.distsAll, dnap.maxDirectDist, dnap.nodesAtmSel)

        # Test creation of a path between two identical nodes
        # Should just return without creating any representation
        viewPath(w, [optpath[0], optpath[0]],
                 dnap.distsAll, dnap.maxDirectDist, dnap.nodesAtmSel)
