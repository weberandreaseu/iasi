import logging
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset, Group, Variable
from iasi.quadrant import Quadrant


class DecompositionException(Exception):
    pass


class Decomposition:

    @staticmethod
    def factory(variable: Variable):
        if variable.name.endswith('atm_n'):
            return EigenDecomposition(variable)
        if variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return SingularValueDecomposition(variable)
        raise DecompositionException(f'Variable {variable.name} cannot be decomposed')

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
    def __init__(self, variable: Variable):
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(var)
        events, lower_bound, upper_bound = q.transformed_shape()
        # tranformed shape 1: (e, gl, gl), 2: (e, gl, 2*gl), 4:(e, 2* gl, 2*gl)
        all_U = np.ma.masked_all((events, lower_bound, lower_bound))
        all_s = np.ma.masked_all((events, lower_bound))
        all_Vh = np.ma.masked_all((events, lower_bound, upper_bound))
        for event in range(var.shape[0]):
            level = levels[event]
            if np.ma.is_masked(level):
                continue
            # reduce array dimensions
            matrix = q.transform(var[event][...], level)
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
        upper_dim, lower_dim = q.upper_and_lower_dimension()
        path = f'{group.path}/{var.name}/'
        U_dim = ('event', lower_dim, lower_dim)
        U_out = output.createVariable(path + 'U', 'f', U_dim, zlib=True)
        U_out[:] = all_U[:]
        s_dim = ('event', lower_dim)
        s_out = output.createVariable(path + 's', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:]
        Vh_dim = ('event', lower_dim, upper_dim)
        Vh_out = output.createVariable(path + 'Vh', 'f', Vh_dim, zlib=True)
        Vh_out[:] = all_Vh[:]


class EigenDecomposition(Decomposition):
    def __init__(self, variable: Variable):
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(var)
        events, _, bound = q.transformed_shape()
        # should be always the same because reshaped variable is square
        all_Q = np.ma.masked_all((events, bound, bound))
        all_s = np.ma.masked_all((events, bound))
        for event in range(var.shape[0]):
            level = levels[event]
            if np.ma.is_masked(level):
                continue
            # reduce array dimensions
            matrix = q.transform(var[event][...], level)
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
        dimension_name, _ = q.upper_and_lower_dimension()
        path = f'{group.path}/{var.name}/'
        Q_dim = ('event', dimension_name, dimension_name)
        Q_out = output.createVariable(path + '/Q', 'f', Q_dim, zlib=True)
        Q_out[:] = all_Q[:]
        s_dim = ('event', dimension_name)
        s_out = output.createVariable(path + '/s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:]
