import logging as logger
import os
from math import ceil
from typing import Dict, List, Tuple

import luigi
import numpy as np
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Group, Variable
from scipy import linalg

from iasi.decomposition import Decomposition, DecompositionException
from iasi.composition import Composition, CompositionException
from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.quadrant import Quadrant
from iasi.util import child_groups_of, child_variables_of


class CompressionParams(luigi.Config):
    threshold = luigi.FloatParameter(default=1e-3)


@requires(MoveVariables)
class CompressDataset(CompressionParams, CopyNetcdfFile):

    exclusion_pattern = r"state"

    def output(self):
        return self.create_local_target('compression', str(self.threshold), file=self.file)

    def run(self):
        input = Dataset(self.input().path)
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        levels = input['atm_nol'][...]
        dim_levels = input.dimensions['atmospheric_grid_levels'].size
        dim_species = input.dimensions['atmospheric_species'].size
        output.createGroup('state')
        output.createDimension(
            'double_atmospheric_grid_levels', dim_levels * dim_species)
        for group, var in child_variables_of(input['state']):
            try:
                dec = Decomposition.factory(var, self.threshold)
                dec.decompose(output, group, var, levels,
                              dim_species, dim_levels)
            except DecompositionException:
                self.copy_variable(output, var, group.path)
        input.close()
        output.close()


@requires(CompressDataset)
class DecompressDataset(CompressionParams, CopyNetcdfFile):

    exclusion_pattern = r"state"

    def output(self):
        return self.create_local_target('decompression', str(self.threshold), file=self.file)

    def run(self):
        input = Dataset(self.input().path)
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        levels = input['atm_nol'][...]
        for group in child_groups_of(input['state']):
            try:
                comp = Composition.factory(group)
                comp.reconstruct(levels, output)
            except CompositionException:
                for var in group.variables.values():
                    self.copy_variable(output, var, group.path)
        input.close()
        output.close()


class SelectSingleVariable(CompressionParams, CopyNetcdfFile):
    gas = luigi.Parameter(default=None)
    variable = luigi.Parameter()
    ancestor = luigi.Parameter(default='MoveVariables')

    def requires(self):
        if self.ancestor == 'MoveVariables':
            return MoveVariables(dst=self.dst, file=self.file)
        if self.ancestor == 'CompressDataset':
            return CompressDataset(
                dst=self.dst,
                file=self.file,
                threshold=self.threshold
            )
        if self.ancestor == "DecompressDataset":
            return DecompressDataset(
                dst=self.dst,
                file=self.file,
                threshold=self.threshold
            )
        raise ValueError(
            f'Undefined ancestor {self.ancestor} for variable selection')

    def output(self):
        var = f'{self.gas}/{self.variable}' if self.gas else self.variable
        if self.ancestor == 'MoveVariables':
            return self.create_local_target('single', 'original', var, str(self.threshold), file=self.file)
        if self.ancestor == 'CompressDataset':
            return self.create_local_target('single', 'compressed', var, str(self.threshold), file=self.file)
        if self.ancestor == 'DecompressDataset':
            return self.create_local_target('single', 'decompressed', var, str(self.threshold), file=self.file)
        raise ValueError(f'Undefined ancestor {self.ancestor}')

    def run(self):
        input = Dataset(self.input().path, 'r')
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        # attribute can be netcdf variable or group (in case of decomposition)
        if self.gas:
            var_path = os.path.join('/state', self.gas, self.variable)
        else:
            var_path = self.variable
        attribute = input[var_path]
        if isinstance(attribute, Group):
            for var in attribute.variables.values():
                compressed = self.ancestor == 'CompressDataset'
                self.copy_variable(
                    output, var, attribute.path, compressed=compressed)
        else:
            assert isinstance(attribute, Variable)
            compressed = self.ancestor == 'CompressDataset'
            path, _ = os.path.split(var_path)
            self.copy_variable(output, attribute, path, compressed=compressed)
