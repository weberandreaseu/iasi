import os

import numpy as np
from netCDF4 import Dataset, Group, Variable


class Quadrant:

    matches = ('event', 'atmospheric_grid_levels', 'atmospheric_grid_levels')

    @classmethod
    def for_assembly(cls, variable: Variable):
        # get and initialize quadrant which assembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.matches == dimensions, [cls, AssembleTwoQuadrants, AssembleFourQuadrants]))(variable)

    @classmethod
    def for_disassembly(cls, variable: Variable):
        # get and initialize quadrant which disassembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.matches == dimensions, [cls, DisassembleTwoQuadrants, DisassembleFourQuadrants]))(variable)

    def __init__(self, variable: Variable = None):
        self.var = variable

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def transformed_shape(self):
        return self.var.shape

    def create_variable(self, output: Dataset, path: str) -> Variable:
        return output.createVariable(path, self.var.datatype, self.matches)

    def assign_disassembly(self, of, to, l):
        to[:l, :l] = of[:l, :l]

    def upper_and_lower_dimension(self):
        return ('atmospheric_grid_levels', 'atmospheric_grid_levels')


class AssembleTwoQuadrants(Quadrant):

    matches = ('event', 'atmospheric_species',
               'atmospheric_grid_levels', 'atmospheric_grid_levels')

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return np.block([array[0, :levels, :levels], array[1, :levels, :levels]])

    def transformed_shape(self):
        grid_levels = self.var.shape[3]
        return (self.var.shape[0], grid_levels, grid_levels * 2)

    def create_variable(self, output: Dataset, path: str) -> Variable:
        return output.createVariable(path, self.var.datatype,
                                     DisassembleTwoQuadrants.matches)

    def upper_and_lower_dimension(self):
        return ('double_atmospheric_grid_levels', 'atmospheric_grid_levels')


class DisassembleTwoQuadrants(Quadrant):
    matches = ('event', 'atmospheric_grid_levels',
               'double_atmospheric_grid_levels')

    def transform(self, array: np.ma.MaskedArray, l: int):
        d = self.var.shape[1]
        return np.array([array[:l, :l], array[:l, d:d + l]])

    def transformed_shape(self):
        grid_levels = self.var.shape[1]
        return (self.var.shape[0], 2, grid_levels, grid_levels)

    def create_variable(self, output: Dataset, path: str) -> Variable:
        return output.createVariable(path, self.var.datatype,
                                     AssembleTwoQuadrants.matches)

    def assign_disassembly(self, of, to, l):
        to[0, :l, :l] = of[:l, :l]
        to[1, :l, :l] = of[:l, l:2*l]


class AssembleFourQuadrants(Quadrant):

    matches = ('event', 'atmospheric_species', 'atmospheric_species',
               'atmospheric_grid_levels', 'atmospheric_grid_levels')

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return np.block([
            [array[0, 0, :levels, :levels], array[1, 0, :levels, :levels]],
            [array[0, 1, :levels, :levels], array[1, 1, :levels, :levels]]
        ])

    def transformed_shape(self):
        grid_levels = self.var.shape[4]
        return (self.var.shape[0], grid_levels * 2, grid_levels * 2)

    def upper_and_lower_dimension(self):
        return ('double_atmospheric_grid_levels', 'double_atmospheric_grid_levels')


class DisassembleFourQuadrants(Quadrant):
    matches = ('event', 'double_atmospheric_grid_levels',
               'double_atmospheric_grid_levels')

    def transform(self, a: np.ma.MaskedArray, l: int):
        d = int(self.var.shape[2] / 2)
        return np.array([
            [a[0:l + 0, 0:l + 0], a[d:d + l, 0:l + 0]],
            [a[0:l + 0, d:d + l], a[d:d + l, d:d + l]]
        ])

    def transformed_shape(self):
        grid_levels = int(self.var.shape[2] / 2)
        return (self.var.shape[0], 2, 2, grid_levels, grid_levels)

    def create_variable(self, output: Dataset, path: str) -> Variable:
        return output.createVariable(path, self.var.datatype,
                                     AssembleFourQuadrants.matches)

    def assign_disassembly(self, reconstructed, result, l):
        result[0, 0, :l, :l] = reconstructed[:l, :l]
        result[1, 0, :l, :l] = reconstructed[:l, l:2*l]
        result[0, 1, :l, :l] = reconstructed[l:2*l, :l]
        result[1, 1, :l, :l] = reconstructed[l:2*l, l:2*l]
