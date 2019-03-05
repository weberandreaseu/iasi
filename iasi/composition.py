import logging

import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.quadrant import Quadrant


class Composition:
    @staticmethod
    def factory(group: Group):
        if 'U' in group.variables.keys():
            return SingularValueComposition(group)
        if 'Q' in group.variables.keys():
            return EigenComposition(group)
        raise ValueError('Group {} cannot be composed'.format(group.name))

    def __init__(self, group: Group):
        self.group = group

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None):
        raise NotImplementedError

    def _export_reconstruction(self, target: Dataset, array: np.ma.MaskedArray, quadrant: Quadrant):
        var = quadrant.create_variable(target, self.group.path)
        var[:] = array[:]


class SingularValueComposition(Composition):

    def __init__(self, group: Group):
        super().__init__(group)
        vars = group.variables.keys()
        assert 'U' in vars and 's' in vars and 'Vh' in vars
        self.U = group['U']
        self.s = group['s']
        self.Vh = group['Vh']

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_disassembly(self.Vh)
        result = np.ma.masked_all(q.transformed_shape(), dtype=np.float32)
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
        if target:
            self._export_reconstruction(target, result, q)
        return result


class EigenComposition(Composition):

    def __init__(self, group: Group):
        super().__init__(group)
        vars = group.variables.keys()
        assert 'Q' in vars and 's' in vars
        self.Q = group['Q']
        self.s = group['s']

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_disassembly(self.Q)
        result = np.ma.masked_all(q.transformed_shape(), dtype=np.float32)
        for event in range(self.Q.shape[0]):
            if np.ma.is_masked(nol[event]):
                logging.warning('Skipping event %d', event)
            level = nol.data[event]
            Q = self.Q[event][...]
            s = self.s[event][...]
            reconstruction = (Q * s).dot(Q.T)
            q.assign_disassembly(reconstruction, result[event], level)
            # result[event] = q.disassemble(reconstruction, nol[event])
        if target:
            self._export_reconstruction(target, result, q)
        return result
