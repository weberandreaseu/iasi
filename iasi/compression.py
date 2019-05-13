import logging
import os
from math import ceil
from typing import Dict, List, Tuple

import luigi
import numpy as np
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Group, Variable
from scipy import linalg

from iasi.composition import Composition, CompositionException
from iasi.decomposition import Decomposition, DecompositionException
from iasi.file import CopyNetcdfFile, MoveVariables, ReadFile
from iasi.quadrant import Quadrant
from iasi.util import child_groups_of, child_variables_of

logger = logging.getLogger(__name__)


class CompressionParams(luigi.Config):
    threshold = luigi.FloatParameter(default=None)


class CompressDataset(CompressionParams, CopyNetcdfFile):

    def requires(self):
        return MoveVariables(file=self.file, dst=self.dst)

    def output_directory(self):
        if self.threshold:
            return os.path.join('compression', str(self.threshold))
        return 'compression'

    def run(self):
        input = Dataset(self.input().path)
        with self.output().temporary_path() as target:
            output = Dataset(target, 'w', format=self.format)
            self.copy_dimensions(input, output, recursive=False)
            # exclude variables starting with state
            self.copy_variables(
                input, output, exclusion_pattern=r'\/?state\S*')
            # TODO refactor
            levels = input['atm_nol'][...]
            dim_levels = input.dimensions['atmospheric_grid_levels'].size
            dim_species = input.dimensions['atmospheric_species'].size
            output.createGroup('state')
            output.createDimension(
                'double_atmospheric_grid_levels', dim_levels * dim_species)
            variables = list(child_variables_of(input['state']))
            counter = 1
            for group, var in variables:
                message = f'Compressing variable {counter} of {len(variables)}: {group.path}/{var.name}'
                self.set_status_message(message)
                progress = int((counter / len(variables)) * 100)
                self.set_progress_percentage(progress)
                try:
                    dec = Decomposition.factory(group, var, self.threshold)
                    logger.info(f'Decompose {group.path}/{var.name}')
                    dec.decompose(output, levels)
                except DecompositionException:
                    logger.debug(
                        f'{group.path}/{var.name} cannot be decomposed. copy without compression')
                    self.copy_variable(output, var, f'{group.path}/{var.name}')
                counter += 1
            input.close()
            output.close()


class DecompressDataset(CompressionParams, CopyNetcdfFile):

    compress_upstream = luigi.BoolParameter(default=False)

    def requires(self):
        if self.compress_upstream:
            return CompressDataset(file=self.file, dst=self.dst, threshold=self.threshold)
        return ReadFile(file=self.file)

    def output_directory(self):
        if self.threshold:
            return os.path.join('decompression', str(self.threshold))
        return 'decompression'

    def run(self):
        input = Dataset(self.input().path)
        with self.output().temporary_path() as target:
            output = Dataset(target, 'w', format=self.format)
            self.copy_dimensions(input, output, recursive=False)
            # exclude variables starting with state
            self.copy_variables(
                input, output, exclusion_pattern=r'\/?state\S*')
            levels = input['atm_nol'][...]
            groups = list(child_groups_of(input['state']))
            counter = 0
            for group in groups:
                counter += 1
                message = f'Reconstruct group {counter} of {len(groups)}: {group.path}'
                self.set_status_message(message)
                self.set_progress_percentage(int(counter / len(groups) * 100))
                logger.info(message)
                try:
                    comp = Composition.factory(group)
                    comp.reconstruct(levels, output)
                except CompositionException:
                    for var in group.variables.values():
                        self.copy_variable(
                            output, var, path=f'{group.path}/{var.name}')
            input.close()
            output.close()


class SelectSingleVariable(CompressionParams, CopyNetcdfFile):
    gas = luigi.Parameter(default='')
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

    def output_directory(self):
        var = f'{self.gas}/{self.variable}' if self.gas else self.variable
        if self.ancestor == 'MoveVariables':
            return os.path.join('single', 'original', var, str(self.threshold))
        if self.ancestor == 'CompressDataset':
            return os.path.join('single', 'compressed', var, str(self.threshold))
        if self.ancestor == 'DecompressDataset':
            return os.path.join('single', 'decompressed', var, str(self.threshold))
        raise ValueError(f'Undefined ancestor {self.ancestor}')

    def run(self):
        input = Dataset(self.input().path, 'r')
        with self.output().temporary_path() as target:
            output = Dataset(target, 'w', format=self.format)
            self.copy_dimensions(input, output)
            # attribute can be netcdf variable or group (in case of decomposition)
            if self.gas:
                var_path = os.path.join('/state', self.gas, self.variable)
            else:
                var_path = self.variable
            attribute = input[var_path]
            if isinstance(attribute, Group):
                for var in attribute.variables.values():
                    self.copy_variable(
                        output, var, f'{attribute.path}/{var.name}', compressed=True)
            else:
                assert isinstance(attribute, Variable)
                self.copy_variable(output, attribute,
                                   var_path, compressed=True)
