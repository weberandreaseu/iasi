import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import Quadrant


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
        self.Q = group['Q']
        self.s = group['s']

    def reconstruct(self, nol: np.ma.MaskedArray) -> np.ma.MaskedArray:
        # determine if variable is composed by quadrants -> dimension of output
        quadrant: Quadrant = Quadrant.for_disassembly(self.Q)
        result = np.ma.masked_all(self.Q.shape, dtype=np.float32)
        # iterate over events
        for event in range(self.Q.shape[0]):
            Q = self.Q[event][...]
            s = self.s[event][...]
            result[event] = (Q * s).dot(Q.T)
        return np.reshape(result, quadrant.disassembly_shape())
