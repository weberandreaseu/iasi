import logging as logger
import os
from math import ceil
from typing import Dict, List, Tuple

import luigi
import numpy as np
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Group, Variable
from scipy import linalg

from iasi.decomposition import Decomposition
from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import child_groups_of, child_variables_of


@requires(MoveVariables)
class CompressDataset(CopyNetcdfFile):

    exclusion_pattern = r"state"

    dimension_names = {}

    def output(self):
        return self.create_local_target('compression', file=self.file)

    def run(self):
        input = Dataset(self.input().path)
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        levels = input['atm_nol'][...]
        dim_levels = input.dimensions['atmospheric_grid_levels'].size
        dim_species = input.dimensions['atmospheric_species'].size
        state = output.createGroup('state')
        state.createDimension(
            'double_atmospheric_grid_levels', dim_levels * dim_species)
        self.dimension_names[dim_species] = 'atmospheric_species'
        self.dimension_names[dim_levels *
                             dim_species] = 'double_atmospheric_grid_levels'
        self.dimension_names[dim_levels] = 'atmospheric_grid_levels'
        for group, var in child_variables_of(input['state']):
            try:
                dec = Decomposition(var, self.dimension_names)
                dec.decompose(output, group, var, levels,
                              dim_species, dim_levels)
            except ValueError:
                self.copy_variable(output, var, group.path)
        input.close()
        output.close()

        # @requires(MoveVariables)
        # class CompressionTask(CopyNetcdfFile):
        #     dim = luigi.IntParameter()
        #     exclusion_pattern = r"state"

        #     def output(self):
        #         self.create_local_target('compression', file=self.file)

        #     def run(self):
        #         input = Dataset(self.input().path)
        #         output = Dataset(self.output().path, 'w', format=self.format)
        #         output.createDimension('kernel_eigenvalues', self.dim)
        #         self.copy_dimensions(input, output)
        #         self.copy_variables(input, output)
        #         # get relevant dimensions
        #         events = input.dimensions['event'].size
        #         grid_levels = input.dimensions['atmospheric_grid_levels'].size
        #         species = input.dimensions['atmospheric_species'].size
        #         # create three components
        #         self.create_variables(output, grid_levels)
        #         avk = input['/state/WV/atm_avk'][...]
        #         nol = input['atm_nol'][...]
        #         # decompose average kernel for each event
        #         for event in range(events):
        #             for row in range(species):
        #                 for column in range(species):
        #                     levels = nol[event]
        #                     # TODO: clarify when input is assumed as valid
        #                     if np.ma.is_masked(levels):
        #                         continue
        #                     levels = int(levels)
        #                     kernel = avk[event, row, column, :levels, :levels]
        #                     if np.ma.is_masked(kernel):
        #                         continue
        #                     # reduce dimension to given number of eigenvalues
        #                     # max number of eigenvalues is number of levels
        #                     dim = min(self.dim, levels)
        #                     if np.isnan(kernel.data).any() or np.isinf(kernel.data).any():
        #                         continue
        #                     self.decompose(output, kernel.data, event,
        #                                    row, column, levels, dim)
        #         input.close()
        #         output.close()

        #     def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        #         raise NotImplementedError

        # class EigenDecomposition(CompressionTask):

        #     def output(self):
        #         return self.create_local_target('eigen', str(self.dim), file=self.file)

        #     def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        #         eigenvalues, eigenvectors = np.linalg.eig(kernel)
        #         # TODO: add warnings
        #         # make sure imaginary part for first eigenvalues is near null
        #         # np.testing.assert_allclose(np.imag(eigenvalues[:dim]), 0)
        #         Q = np.real(eigenvectors[:, :dim])
        #         s = np.real(eigenvalues[:dim])
        #         output['/state/WV/atm_avk/Q'][event, row, column, :levels, :dim] = Q
        #         output['/state/WV/atm_avk/s'][event, row, column, :dim] = s

        #     def create_variables(self, dataset: Dataset, grid_levels: int) -> None:
        #         Q = dataset.createVariable('/state/WV/atm_avk/Q', 'f8',
        #                                    ('event', 'atmospheric_species', 'atmospheric_species',
        #                                     'atmospheric_grid_levels', 'kernel_eigenvalues'),
        #                                    fill_value=-9999.9)
        #         Q.description = "Eigenvectors of eigen decomposition"
        #         s = dataset.createVariable('/state/WV/atm_avk/s', 'f8',
        #                                    ('event', 'atmospheric_species',
        #                                     'atmospheric_species', 'kernel_eigenvalues'),
        #                                    fill_value=-9999.9)
        #         s.description = "Eigenvalues (sigma) of eigen decomposition"

        # class SingularValueDecomposition(CompressionTask):

        #     def output(self):
        #         return self.create_local_target('svd', str(self.dim), file=self.file)

        #     def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        #         U, s, Vh = linalg.svd(kernel.data, full_matrices=False)
        #         U = U[:, :dim]
        #         s = s[:dim]
        #         Vh = Vh[:dim, :]
        #         output['/state/WV/atm_avk/U'][event, row,
        #                                       column, : levels, : dim] = U
        #         output['/state/WV/atm_avk/s'][event, row, column, : dim] = s
        #         output['/state/WV/atm_avk/Vh'][event, row,
        #                                        column, : dim, : levels] = Vh

        #     def create_variables(self, dataset: Dataset, grid_levels: int) -> None:
        #         U = dataset.createVariable('/state/WV/atm_avk/U', 'f8',
        #                                    ('event', 'atmospheric_species', 'atmospheric_species',
        #                                     'atmospheric_grid_levels', 'kernel_eigenvalues'),
        #                                    fill_value=-9999.9)
        #         U.description = "U component of singular value decomposition"
        #         s = dataset.createVariable('/state/WV/atm_avk/s', 'f8',
        #                                    ('event', 'atmospheric_species',
        #                                     'atmospheric_species', 'kernel_eigenvalues'),
        #                                    fill_value=-9999.9)
        #         s.description = "Eigenvalues (sigma) of singular value decomposition"
        #         Vh = dataset.createVariable('/state/WV/atm_avk/Vh', 'f8',
        #                                     ('event', 'atmospheric_species', 'atmospheric_species',
        #                                      'kernel_eigenvalues', 'atmospheric_grid_levels'),
        #                                     fill_value=-9999.9)
        #         Vh.description = "Vh component of singular value decomposition"
