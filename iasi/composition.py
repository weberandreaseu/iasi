import logging

import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import Quadrant


class Composition:

    @staticmethod
    def factory(group: Group):
        if 'U' in group.variables.keys():
            return SingularValueComposition(group)
        if 'Q' in group.variables.keys():
            return EigenComposition(group)
        raise ValueError('Group {} cannot be composed'.format(group.name))

    def reconstruct(self, nol: np.ma.MaskedArray):
        raise NotImplementedError

    def export_reconstruction(self, output: Dataset, nol: np.ma.MaskedArray):
        raise NotImplementedError


class SingularValueComposition:

    def __init__(self, group: Group):
        vars = group.variables.keys()
        assert 'U' in vars and 's' in vars and 'Vh' in vars
        self.group = group
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

    def export_reconstruction(self, output: Dataset, nol: np.ma.MaskedArray):
        matrix = self.reconstruct(nol)
        q: Quadrant = Quadrant.for_disassembly(self.Vh)
        var = output.createVariable(
            self.group.path, self.Vh.datatype, q.assembles)
        var[:] = matrix[:]


class EigenComposition:

    def __init__(self, group: Group):
        vars = group.variables.keys()
        assert 'Q' in vars and 's' in vars
        self.group = group
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

    def export_reconstruction(self, output: Dataset, nol: np.ma.MaskedArray):
        matrix = self.reconstruct(nol)
        q: Quadrant = Quadrant.for_disassembly(self.Q)
        var = output.createVariable(
            self.group.path, self.Q.datatype, q.assembles)
        var[:] = matrix[:]
