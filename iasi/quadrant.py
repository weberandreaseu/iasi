import numpy as np
from netCDF4 import Dataset, Group, Variable


class Quadrant:

    assembles = ('event', 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = assembles

    @classmethod
    def for_assembly(cls, variable: Variable):
        # get and initialize quadrant which assembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.assembles == dimensions, [cls, TwoQuadrants, FourQuadrants]))(variable)

    @classmethod
    def for_disassembly(cls, variable: Variable):
        # get and initialize quadrant which disassembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.disassembles == dimensions, [cls, TwoQuadrants, FourQuadrants]))(variable)

    def __init__(self, variable: Variable):
        self.var = variable

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def disassemble(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def create_variable(self, output: Dataset):
        raise NotImplementedError

    def assembly_shape(self):
        return self.var.shape

    def disassembly_shape(self):
        return self.var.shape

    def assign_disassembly(self, of, to, l):
        to[:l, :l] = of[:l, :l]


class TwoQuadrants(Quadrant):

    assembles = ('event', 'atmospheric_species',
                 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = ('event', 'atmospheric_grid_levels',
                    'double_atmospheric_grid_levels')

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return np.block([array[0, :levels, :levels], array[1, :levels, :levels]])

    def assembly_shape(self):
        grid_levels = self.var.shape[3]
        return (self.var.shape[0], grid_levels, grid_levels * 2)

    def disassembly_shape(self):
        grid_levels = self.var.shape[1]
        return (self.var.shape[0], 2, grid_levels, grid_levels)

    def disassemble(self, array: np.ma.MaskedArray, l: int):
        d = self.var.shape[1]
        return np.array([array[:l, :l], array[:l, d:d + l]])

    def assign_disassembly(self, of, to, l):
        to[0, :l, :l] = of[:l, :l]
        to[1, :l, :l] = of[:l, l:2*l]


class FourQuadrants(Quadrant):

    assembles = ('event', 'atmospheric_species', 'atmospheric_species',
                 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = ('event', 'double_atmospheric_grid_levels',
                    'double_atmospheric_grid_levels')

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return np.block([
            [array[0, 0, :levels, :levels], array[0, 1, :levels, :levels]],
            [array[1, 0, :levels, :levels], array[1, 1, :levels, :levels]]
        ])

    def assembly_shape(self):
        grid_levels = self.var.shape[4]
        return (self.var.shape[0], grid_levels * 2, grid_levels * 2)

    def disassembly_shape(self):
        grid_levels = int(self.var.shape[2] / 2)
        return (self.var.shape[0], 2, 2, grid_levels, grid_levels)

    def disassemble(self, a: np.ma.MaskedArray, l: int):
        d = int(self.var.shape[2] / 2)
        return np.array([
            [a[:l, :l], a[:l, d:d + l]],
            [a[d:d + l, :l], a[d:d + l, d:d + l]]
        ])

    def assign_disassembly(self, of, to, l):
        to[0, 0, :l, :l] = of[:l, :l]
        to[0, 1, :l, :l] = of[:l, l:2*l]
        to[1, 0, :l, :l] = of[l:2*l, :l]
        to[1, 1, :l, :l] = of[l:2*l, l:2*l]

    # def slice(levels: int):
    #     return slice()
