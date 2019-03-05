import logging

import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import Quadrant


class Composition:

    def __init__(self, group: Group):
        if 'U' in group.variables.keys():
            self.composition = SingularValueComposition(group)
            return
        if 'Q' in group.variables.keys():
            self.composition = EigenComposition(group)
            return
        raise ValueError('Group {} cannot be composed'.format(group.name))

    def reconstruct(self, nol: np.ma.MaskedArray):
        return self.composition.reconstruct(nol)


class SingularValueComposition:

    def __init__(self, group: Group):
        vars = group.variables.keys()
        assert 'U' in vars and 's' in vars and 'Vh' in vars
        self.U = group['U']
        self.s = group['s']
        self.Vh = group['Vh']

    def reconstruct(self, nol: np.ma.MaskedArray) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_disassembly(self.Vh)
        result = np.ma.masked_all(q.disassembly_shape(), dtype=np.float32)
        for event in range(self.Vh.shape[0]):
            if np.ma.is_masked(nol[event]):
                logging.warning('Skipping event %d', event)
            level = nol.data[event]
            U = self.U[event][...]
            s = self.s[event][...]
            Vh = self.Vh[event][...]
            reconstruction = (U * s).dot(Vh)
            q.assign_disassembly(reconstruction, result[event], level)
            # result[event] = q.disassemble(reconstruction, nol[event])
        return result


class EigenComposition:

    def __init__(self, group: Group):
        vars = group.variables.keys()
        assert 'Q' in vars and 's' in vars
        self.Q = group['Q']
        self.s = group['s']

    def reconstruct(self, nol: np.ma.MaskedArray) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_disassembly(self.Q)
        result = np.ma.masked_all(q.disassembly_shape(), dtype=np.float32)
        for event in range(self.Q.shape[0]):
            if np.ma.is_masked(nol[event]):
                logging.warning('Skipping event %d', event)
            level = nol.data[event]
            Q = self.Q[event][...]
            s = self.s[event][...]
            reconstruction = (Q * s).dot(Q.T)
            q.assign_disassembly(reconstruction, result[event], level)
            # result[event] = q.disassemble(reconstruction, nol[event])
        return result
