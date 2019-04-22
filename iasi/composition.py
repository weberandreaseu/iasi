import logging

import numpy as np
from luigi.util import requires
from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.quadrant import Quadrant


logger = logging.getLogger(__name__)


class CompositionException(Exception):
    pass


class Composition:
    @staticmethod
    def factory(group: Group):
        if 'U' in group.variables.keys():
            return SingularValueComposition(group)
        if 'Q' in group.variables.keys():
            return EigenComposition(group)
        raise CompositionException(
            'Group {} cannot be composed'.format(group.name))

    def __init__(self, group: Group):
        self.group = group

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None):
        raise NotImplementedError

    def _export_reconstruction(self, target: Dataset, array: np.ma.MaskedArray, quadrant: Quadrant):
        pass
        # TODO implement
        # var = quadrant.create_variable(target, self.group.path)
        # var[:] = array[:]


class SingularValueComposition(Composition):

    def __init__(self, group: Group):
        super().__init__(group)
        vars = group.variables.keys()
        assert 'U' in vars and 's' in vars and 'Vh' in vars
        self.U = group['U']
        self.s = group['s']
        self.Vh = group['Vh']
        self.quadrant = Quadrant.for_disassembly(group.parent.name, group.name, self.U)

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None) -> np.ma.MaskedArray:
        result = np.ma.masked_all(
            self.quadrant.transformed_shape(), dtype=np.float32)
        for event in range(self.Vh.shape[0]):
            if np.ma.is_masked(nol[event]) or nol.data[event] > 29:
                logger.warning('Skipping event %d', event)
                continue
            level = int(nol.data[event])
            U = self.U[event][...]
            s = self.s[event][...]
            Vh = self.Vh[event][...]
            reconstruction = (U * s).dot(Vh)
            self.quadrant.assign_disassembly(
                reconstruction, result[event], level)
            # result[event] = q.disassemble(reconstruction, nol[event])
        if target:
            self._export_reconstruction(target, result, self.quadrant)
        return result


class EigenComposition(Composition):

    def __init__(self, group: Group):
        super().__init__(group)
        vars = group.variables.keys()
        assert 'Q' in vars and 's' in vars
        self.Q = group['Q']
        self.s = group['s']
        self.quadrant = Quadrant.for_disassembly(group.parent.name, group.name, self.Q)

    def reconstruct(self, nol: np.ma.MaskedArray, target: Dataset = None) -> np.ma.MaskedArray:
        result = np.ma.masked_all(
            self.quadrant.transformed_shape(), dtype=np.float32)
        for event in range(self.Q.shape[0]):
            if np.ma.is_masked(nol[event]) or nol.data[event] > 29:
                logger.warning('Skipping event %d', event)
                continue
            level = int(nol.data[event])
            Q = self.Q[event][...]
            s = self.s[event][...]
            reconstruction = (Q * s).dot(Q.T)
            self.quadrant.assign_disassembly(
                reconstruction, result[event], level)
        if target:
            self._export_reconstruction(target, result, self.quadrant)
        return result
