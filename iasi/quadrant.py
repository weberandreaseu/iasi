import os

import numpy as np
from netCDF4 import Dataset, Group, Variable
from iasi.util import dimensions_of


class QuadrantException(Exception):
    pass


# Combinations

#           quadrant    shape
# WV avk    4           (1, 2, 2, 29, 29)   (1, 58, 58)
# WV n      4           (1, 2, 2, 29, 29)   (1, 58, 58)
# WV xavk   2           (1, 2, 29, 29)      (1, 58, 29)

# GHG avk   4           (1, 2, 2, 29, 29)   (1, 58, 58)
# GHG n     4           (1, 2, 2, 29, 29)   (1, 58, 58)
# GHG xavk  2           (1, 2, 29, 29)      (1, 58, 29)

# HNO3 avk  1           (1, 29, 29)         (1, 29, 29)
# HNO3 n    1           (1, 29, 29)         (1, 29, 29)
# HNO3 xavk 1           (1, 29, 29)         (1, 29, 29)

# Tatm avk  1           (1, 29, 29)         (1, 29, 29)
# Tatm n    1           (1, 29, 29)         (1, 29, 29)

class Quadrant:

    @classmethod
    def for_assembly(cls, gas: str,  var_name: str, var: Variable):
        try:
            assert var_name in ['avk', 'n', 'Tatmxavk']
        except AssertionError:
            print(var_name)
        dimensions = dimensions_of(var)
        events = dimensions['event']
        levels = dimensions['atmospheric_grid_levels']
        # single quadrant
        if gas == 'Tatm' or gas == 'HNO3':
            return cls(events, levels)
        if gas == 'WV' or gas == 'GHG':
            # two quadrants
            if var_name == 'Tatmxavk':
                return AssembleTwoQuadrants(events, levels)
            # four quadrants
            else:
                return AssembleFourQuadrants(events, levels)
        raise QuadrantException(f'Cannot find assembly function for {gas}')

    @classmethod
    def for_disassembly(cls, gas: str,  var_name: str, var: Variable):
        # single quadrant
        assert var_name in ['avk', 'n', 'Tatmxavk']
        dimensions = dimensions_of(var)
        events = dimensions['event']
        if 'atmospheric_grid_levels' in dimensions.keys():
            levels = dimensions['atmospheric_grid_levels']
        else:
            levels = int(dimensions['double_atmospheric_grid_levels'] / 2)
        if gas == 'Tatm' or gas == 'HNO3':
            return cls(events, levels)
        if gas == 'WV' or gas == 'GHG':
            # two quadrants
            if var_name == 'Tatmxavk':
                return DisassembleTwoQuadrants(events, levels)
            # four quadrants
            else:
                return DisassembleFourQuadrants(events, levels)
        raise QuadrantException(
            f'Cannot find assembly function for {gas}')

    def __init__(self, events: int, grid_levels=29):
        self.events = events
        self.grid_levels = grid_levels

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def transformed_shape(self):
        return (self.events, self.grid_levels, self.grid_levels)

    def assign_disassembly(self, of, to, l):
        to[:l, :l] = of[:l, :l]

    def upper_and_lower_dimension(self):
        return ('atmospheric_grid_levels', 'atmospheric_grid_levels')


class AssembleTwoQuadrants(Quadrant):

    matches = ('event', 'atmospheric_species',
               'atmospheric_grid_levels', 'atmospheric_grid_levels')

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return np.block([
            [array[0, :levels, :levels]],
            [array[1, :levels, :levels]]
        ])

    def transformed_shape(self):
        return (self.events, self.grid_levels * 2, self.grid_levels)

    def upper_and_lower_dimension(self):
        return ('double_atmospheric_grid_levels', 'atmospheric_grid_levels')


class DisassembleTwoQuadrants(Quadrant):

    def transform(self, array: np.ma.MaskedArray, l: int):
        return np.array([array[:l, :l], array[self.grid_levels:self.grid_levels + l, :l]])

    def transformed_shape(self):
        return (self.events, 2, self.grid_levels, self.grid_levels)

    def assign_disassembly(self, of, to, l):
        to[0, :l, :l] = of[:l, :l]
        to[1, :l, :l] = of[l:2*l, :l]


class AssembleFourQuadrants(Quadrant):

    def transform(self, array: np.ma.MaskedArray, levels: int):
        return np.block([
            [array[0, 0, :levels, :levels], array[1, 0, :levels, :levels]],
            [array[0, 1, :levels, :levels], array[1, 1, :levels, :levels]]
        ])

    def transformed_shape(self):
        return (self.events, self.grid_levels * 2, self.grid_levels * 2)

    def upper_and_lower_dimension(self):
        return ('double_atmospheric_grid_levels', 'double_atmospheric_grid_levels')


class DisassembleFourQuadrants(Quadrant):

    def transform(self, a: np.ma.MaskedArray, l: int):
        d = self.grid_levels
        return np.array([
            [a[0:l + 0, 0:l + 0], a[d:d + l, 0:l + 0]],
            [a[0:l + 0, d:d + l], a[d:d + l, d:d + l]]
        ])

    def transformed_shape(self):
        return (self.events, 2, 2, self.grid_levels, self.grid_levels)

    def assign_disassembly(self, reconstructed, result, l):
        result[0, 0, :l, :l] = reconstructed[:l, :l]
        result[1, 0, :l, :l] = reconstructed[:l, l:2*l]
        result[0, 1, :l, :l] = reconstructed[l:2*l, :l]
        result[1, 1, :l, :l] = reconstructed[l:2*l, l:2*l]
