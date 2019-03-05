import logging
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset, Group, Variable


class Decomposition:

    @staticmethod
    def factory(variable: Variable, dimension_names):
        if variable.name.endswith('atm_n'):
            return EigenDecomposition(variable, dimension_names)
        if variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return SingularValueDecomposition(variable, dimension_names)
        raise ValueError(f'Variable {variable.name} cannot be decomposed')

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        raise NotImplementedError
        
    def select_significant(self, eigenvalues: List, thres=10e-4) -> List:
        most_significant = eigenvalues[0]
        return list(filter(lambda eig: eig > most_significant * thres, eigenvalues))

    def matrix_ok(self, event, var, matrix):
        ok = True
        if np.ma.is_masked(matrix):
            logging.warning(
                'event %d contains masked values in %s. skipping', event, var.name)
            ok = False
        if np.isnan(matrix).any():
            logging.warning(
                'event %d contains nan values in %s. skipping', event, var.name)
            ok = False
        if np.isinf(matrix).any():
            logging.warning(
                'event %d contains inf values in %s. skipping', event, var.name)
            ok = False
        return ok


class SingularValueDecomposition(Decomposition):
    def __init__(self, variable: Variable, dimension_names):
        self.dimension_names = dimension_names
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        reshaper = ArrayReshaper(var.shape, dim_levels, dim_species)
        events, upper, lower = reshaper.new_shape()

        all_U = np.ma.masked_all((events, lower, lower))
        all_s = np.ma.masked_all((events, lower))
        # TODO (upper, upper) or (lower, upper)?
        all_Vh = np.ma.masked_all((events, lower, upper))
        for event in range(var.shape[0]):
            level = levels[event]
            if np.ma.is_masked(level):
                continue
            # reduce array dimensions
            matrix = reshaper.reshape(var[event][...], level)
            if not self.matrix_ok(event, var, matrix):
                continue
            # decompose reduced array
            U, s, Vh = np.linalg.svd(matrix.data, full_matrices=False)
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


class EigenDecomposition(Decomposition):
    def __init__(self, variable: Variable, dimension_names):
        self.dimension_names = dimension_names
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
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
                logging.error('Failed to assign values')
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
