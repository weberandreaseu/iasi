import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables


@requires(MoveVariables)
class Compositon(CopyNetcdfFile):
    pass


class SingularValueComposition:

    def __init__(self, group: Group):
        # assert group contains U, s, VH
        pass


class EigenCompositon:

    def __init__(self, group: Group):
        vars = group.variables.keys()
        assert 'Q' in vars and 's' in vars
        # if 'double_atmospheric_grid_levels' in group['Q'].dimenisons:
        #     pass

        self.Q = group['Q']
        self.s = group['s'][...]

    def reconstruct(self, nol: np.ma.MaskedArray) -> np.ndarray:
        print(self.Q.dimensions)
        # determine if variable is composed by quadrants -> dimension of output
        # allocate ndarray with shape
        # iterate over events
        # decompose matrix for each event
        # if quadrants, get 4 quadrants
