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


@requires(MoveVariables)
class CompressDataset(CopyNetcdfFile):

    exclusion_pattern = r"state"
    thres_eigenvalues = luigi.FloatParameter(default=1e-3)

    def output(self):
        return self.create_local_target('compression', str(self.thres_eigenvalues), file=self.file)

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
                dec = Decomposition.factory(var, self.thres_eigenvalues)
                dec.decompose(output, group, var, levels,
                              dim_species, dim_levels)
            except DecompositionException:
                self.copy_variable(output, var, group.path)
        input.close()
        output.close()


@requires(CompressDataset)
class DecompressDataset(CopyNetcdfFile):

    exclusion_pattern = r"state"

    def output(self):
        return self.create_local_target('decompression', file=self.file)

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


class SelectSingleVariable(CopyNetcdfFile):
    variable = luigi.Parameter()
    compressed = luigi.BoolParameter()

    def requires(self):
        if self.compressed:
            return CompressDataset(dst=self.dst, file=self.file)
        else:
            return MoveVariables(dst=self.dst, file=self.file)

    def output(self):
        type = 'compressed' if self.compressed else 'uncompressed'
        return self.create_local_target('single', type, self.variable, file=self.file)

    def run(self):
        input = Dataset(self.input().path, 'r')
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        # attribute can be netcdf variable or group (in case of decomposition)
        attribute = input[self.variable]
        if isinstance(attribute, Group):
            for var in attribute.variables.values():
                self.copy_variable(output, var, attribute.path)
        else:
            assert isinstance(attribute, Variable)
            path, _ = os.path.split(self.variable)
            self.copy_variable(output, attribute, path)

