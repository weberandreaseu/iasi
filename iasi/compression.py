import logging as logger
import os
from math import ceil
from typing import Dict, List, Tuple

import luigi
import numpy as np
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Group, Variable
from scipy import linalg

from iasi.file import CopyNetcdfFile, MoveVariables


@requires(MoveVariables)
class CompressionTask(CopyNetcdfFile):
    dim = luigi.IntParameter()
    exclusion_pattern = r"state"

    def output(self):
        self.create_local_target('compression', file=self.file)

    def run(self):
        input = Dataset(self.input().path)
        output = Dataset(self.output().path, 'w', format=self.format)
        output.createDimension('kernel_eigenvalues', self.dim)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        # get relevant dimensions
        events = input.dimensions['event'].size
        grid_levels = input.dimensions['atmospheric_grid_levels'].size
        species = input.dimensions['atmospheric_species'].size
        # create three components
        self.create_variables(output, grid_levels)
        avk = input['/state/WV/atm_avk'][...]
        nol = input['atm_nol'][...]
        # decompose average kernel for each event
        for event in range(events):
            for row in range(species):
                for column in range(species):
                    levels = nol[event]
                    # TODO: clarify when input is assumed as valid
                    if np.ma.is_masked(levels):
                        continue
                    levels = int(levels)
                    kernel = avk[event, row, column, :levels, :levels]
                    if np.ma.is_masked(kernel):
                        continue
                    # reduce dimension to given number of eigenvalues
                    # max number of eigenvalues is number of levels
                    dim = min(self.dim, levels)
                    if np.isnan(kernel.data).any() or np.isinf(kernel.data).any():
                        continue
                    self.decompose(output, kernel.data, event,
                                   row, column, levels, dim)
        input.close()
        output.close()

    def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        raise NotImplementedError


@requires(MoveVariables)
class GroupCompression(CopyNetcdfFile):

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
        child_groups = self.child_groups(input['state'])
        for group in filter(lambda g: g.variables, child_groups):
            for name, var in group.variables.items():
                # for decomposition, array should be of higher dimension
                # last two dimensions should be "atmospheric_grid_levels"
                if var.dimensions[-2:] != ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
                    self.copy_variable(output, var, group.path)
                    continue
                if name.endswith('atm_n'):
                    # noise matrix. suitable for eigen decomposition
                    self.eigen_decomposition(
                        output, group, var, levels, dim_species, dim_levels)
                else:
                    self.singular_value_decomposition(
                        group, var, dim_species, dim_levels, levels, output)
        input.close()
        output.close()

    def child_groups(self, group: Group):
        if group.groups:
            return [group] + [self.child_groups(g) for g in group.groups.values()]
        else:
            return group

    # def child_variables(self, group: Group) -> List[Tuple['str', Variable]]:
    #     return [g.variables.values() for g in self.child_groups(group)]

    def singular_value_decomposition(self, group: Group, var: Variable, dim_species: int, dim_levels: int, nol: np.ma.MaskedArray, output: Dataset) -> np.ma.MaskedArray:
        reshaper = ArrayReshaper(var.shape, dim_levels, dim_species)
        events, upper, lower = reshaper.new_shape()

        all_U = np.ma.masked_all((events, lower, lower))
        all_s = np.ma.masked_all((events, lower))
        # TODO (upper, upper) or (lower, upper)?
        all_Vh = np.ma.masked_all((events, lower, upper))
        for event in range(var.shape[0]):
            level = nol[event]
            if np.ma.is_masked(level):
                continue
            # reduce array dimensions
            matrix = reshaper.reshape(var[event][...], level)
            if not self.matrix_ok(event, var, matrix):
                continue
            # decompose reduced array
            U, s, Vh = linalg.svd(matrix.data, full_matrices=False)
            # find k eigenvalues
            sigma = self.select_significant(s)
            k = len(sigma)
            # assign sliced decomposition to all
            try:
                all_U[event][:U.shape[0], :k] = U[:, :k]
                all_s[event][:k] = sigma
                all_Vh[event][:k, :Vh.shape[1]] = Vh[:k, :]
            except ValueError as error:
                print(error)
        # write all to output
        U_dim = (
            'event', self.dimension_names[lower], self.dimension_names[lower])
        U_out = output.createVariable(
            f'{group.path}/{var.name}/U', 'f', U_dim, zlib=True)
        U_out[:] = all_U[:]
        s_dim = ('event', self.dimension_names[lower])
        s_out = output.createVariable(
            f'{group.path}/{var.name}/s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:]
        Vh_dim = (
            'event', self.dimension_names[lower], self.dimension_names[upper])
        Vh_out = output.createVariable(
            f'{group.path}/{var.name}/Vh', 'f', Vh_dim, zlib=True)
        Vh_out[:] = all_Vh[:]

    def select_significant(self, eigenvalues: List, thres=10e-4) -> List:
        most_significant = eigenvalues[0]
        return list(filter(lambda eig: eig > most_significant * thres, eigenvalues))

    def matrix_ok(self, event, var, matrix):
        ok = True
        if np.ma.is_masked(matrix):
            logger.warning(
                'event %d contains masked values in %s. skipping', event, var.name)
            ok = False
        if np.isnan(matrix).any():
            logger.warning(
                'event %d contains nan values in %s. skipping', event, var.name)
            ok = False
        if np.isinf(matrix).any():
            logger.warning(
                'event %d contains inf values in %s. skipping', event, var.name)
            ok = False
        return ok

    def eigen_decomposition(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        reshaper = ArrayReshaper(var.shape, dim_levels, dim_species)
        # should be always the same because reshaped variable is square
        events, matrix_size, _ = reshaper.new_shape()
        all_Q = np.ma.masked_all((events, matrix_size, matrix_size))
        all_s = np.ma.masked_all((events, matrix_size))
        for event in range(var.shape[0]):
            level = levels[event]
            if np.ma.is_masked(level):
                continue
            # reduce array dimensions
            matrix = reshaper.reshape(var[event][...], level)
            # decompose reduced array
            if not self.matrix_ok(event, var, matrix):
                continue
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            most_significant = self.select_significant(eigenvalues)
            k = len(most_significant)
            # TODO check for imag values or symmetric property. already happend!
            try:
                all_Q[event][:eigenvectors.shape[0], :k] = eigenvectors[:, :k]
                all_s[event][:k] = most_significant
            except ValueError as error:
                logger.error('Failed to assign values')
        # write all to output
        dimension_name = self.dimension_names[matrix_size]
        Q_dim = ('event', dimension_name, dimension_name)
        Q_out = output.createVariable(
            f'{group.path}/{var.name}/Q', 'f', Q_dim, zlib=True)
        Q_out[:] = all_Q[:]
        s_dim = ('event', dimension_name)
        s_out = output.createVariable(
            f'{group.path}/{var.name}/s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:]


class ArrayReshaper:

    transformer = {
        3: lambda x, l: x[:l, :l],
        4: lambda x, l: np.block([x[0, :l, :l], x[1, :l, :l]]),
        5: lambda x, l: np.block([
            [x[0, 0, :l, :l], x[0, 1, :l, :l]],
            [x[1, 0, :l, :l], x[1, 1, :l, :l]]
        ])
    }

    def __init__(self, shape: Tuple, levels: int, species: int):
        self.shape = shape
        self.levels = levels
        self.species = species

    def new_shape(self) -> Tuple:
        events = self.shape[0]
        dim = len(self.shape)
        if dim == 3:
            return events, self.levels, self.levels
        if dim == 4:
            return events, self.species * self.levels, self.levels
        if dim == 5:
            return events, self.species * self.levels, self.species * self.levels
        raise ValueError('Unexpected shape')

    def reshape(self, array: np.ma.MaskedArray, nol: int) -> np.ndarray:
        transformer = self.transformer.get(len(self.shape))
        return transformer(array, nol)


class EigenDecomposition(CompressionTask):

    def output(self):
        return self.create_local_target('eigen', str(self.dim), file=self.file)

    def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        eigenvalues, eigenvectors = np.linalg.eig(kernel)
        # TODO: add warnings
        # make sure imaginary part for first eigenvalues is near null
        # np.testing.assert_allclose(np.imag(eigenvalues[:dim]), 0)
        Q = np.real(eigenvectors[:, :dim])
        s = np.real(eigenvalues[:dim])
        output['/state/WV/atm_avk/Q'][event, row, column, :levels, :dim] = Q
        output['/state/WV/atm_avk/s'][event, row, column, :dim] = s

    def create_variables(self, dataset: Dataset, grid_levels: int) -> None:
        Q = dataset.createVariable('/state/WV/atm_avk/Q', 'f8',
                                   ('event', 'atmospheric_species', 'atmospheric_species',
                                    'atmospheric_grid_levels', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        Q.description = "Eigenvectors of eigen decomposition"
        s = dataset.createVariable('/state/WV/atm_avk/s', 'f8',
                                   ('event', 'atmospheric_species',
                                    'atmospheric_species', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        s.description = "Eigenvalues (sigma) of eigen decomposition"


class SingularValueDecomposition(CompressionTask):

    def output(self):
        return self.create_local_target('svd', str(self.dim), file=self.file)

    def decompose(self, output: Dataset, kernel: np.ndarray, event: int, row: int, column: int, levels: int, dim: int):
        U, s, Vh = linalg.svd(kernel.data, full_matrices=False)
        U = U[:, :dim]
        s = s[:dim]
        Vh = Vh[:dim, :]
        output['/state/WV/atm_avk/U'][event, row,
                                      column, : levels, : dim] = U
        output['/state/WV/atm_avk/s'][event, row, column, : dim] = s
        output['/state/WV/atm_avk/Vh'][event, row,
                                       column, : dim, : levels] = Vh

    def create_variables(self, dataset: Dataset, grid_levels: int) -> None:
        U = dataset.createVariable('/state/WV/atm_avk/U', 'f8',
                                   ('event', 'atmospheric_species', 'atmospheric_species',
                                    'atmospheric_grid_levels', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        U.description = "U component of singular value decomposition"
        s = dataset.createVariable('/state/WV/atm_avk/s', 'f8',
                                   ('event', 'atmospheric_species',
                                    'atmospheric_species', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        s.description = "Eigenvalues (sigma) of singular value decomposition"
        Vh = dataset.createVariable('/state/WV/atm_avk/Vh', 'f8',
                                    ('event', 'atmospheric_species', 'atmospheric_species',
                                     'kernel_eigenvalues', 'atmospheric_grid_levels'),
                                    fill_value=-9999.9)
        Vh.description = "Vh component of singular value decomposition"
