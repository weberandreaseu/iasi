import logging
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset, Group, Variable
from iasi.quadrant import Quadrant
from iasi.util import dimensions_of

logger = logging.getLogger(__name__)


class DecompositionException(Exception):
    pass


class Decomposition:

    threshold = 1e-3

    @classmethod
    def factory(cls, variable: Variable, threshold: float = 1e-3):
        cls.threshold = threshold
        # noise matrix (n) is symmetric and is qualified for EigenDecomposition
        if variable.name is 'n' and variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return EigenDecomposition(variable)
        if variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return SingularValueDecomposition(variable)
        raise DecompositionException(
            f'Variable {variable.name} cannot be decomposed')

    # TODO refactor: make methods signature more compact
    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        raise NotImplementedError

    def target_rank(self, eigenvalues: List) -> int:
        most_significant = abs(eigenvalues[0])
        for k, eigenvalue in enumerate(np.abs(eigenvalues)):
            if eigenvalue < most_significant * self.threshold:
                return k
        return len(eigenvalues)

    def matrix_ok(self, event, path, matrix):
        ok = True
        if np.ma.is_masked(matrix):
            logger.warning(
                'event %d contains masked values in %s. skipping...', event, path)
            ok = False
        if np.isnan(matrix).any():
            logger.warning(
                'event %d contains nan values in %s. skipping...', event, path)
            ok = False
        if np.isinf(matrix).any():
            logger.warning(
                'event %d contains inf values in %s. skipping...', event, path)
            ok = False
        return ok


class SingularValueDecomposition(Decomposition):
    def __init__(self, variable: Variable):
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(group.name, var.name, var)
        events, upper_bound, lower_bound = q.transformed_shape()
        # tranformed shape 1: (e, gl, gl), 2: (e, 2*gl, gl), 4:(e, 2* gl, 2*gl)
        all_U = np.ma.masked_all((events, upper_bound, lower_bound))
        all_s = np.ma.masked_all((events, lower_bound))
        all_Vh = np.ma.masked_all((events, lower_bound, lower_bound))
        all_k = np.ma.masked_all((events), dtype=np.int)
        path = f'{group.path}/{var.name}/'
        max_k = 0
        for event in range(var.shape[0]):
            if np.ma.is_masked(levels[event]) or levels.data[event] > 29:
                continue
            # reduce array dimensions
            level = int(levels.data[event])
            matrix = q.transform(var[event][...], level)
            if not self.matrix_ok(event, path, matrix):
                continue
            # decompose reduced array
            U, s, Vh = np.linalg.svd(matrix.data, full_matrices=False)
            # find k eigenvalues
            k = self.target_rank(s)
            sigma = s[:k]
            max_k = max(k, max_k)
            # assign sliced decomposition to all
            all_k[event] = k
            all_U[event][:U.shape[0], :k] = U[:, :k]
            all_s[event][:k] = sigma
            all_Vh[event][:k, :Vh.shape[1]] = Vh[:k, :]
        # write all to output
        upper_dim, lower_dim = q.upper_and_lower_dimension()
        group = output.createGroup(path)
        group.createDimension('rank', size=max_k)
        # TODO add group description
        k_out = group.createVariable('k', 'i1', ('event'))
        k_out[:] = all_k[:]
        U_dim = ('event', upper_dim, 'rank')
        U_out = group.createVariable('U', 'f', U_dim, zlib=True)
        U_out[:] = all_U[:, :, :max_k]
        s_dim = ('event', 'rank')
        s_out = group.createVariable('s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:, :max_k]
        Vh_dim = ('event', 'rank', lower_dim)
        Vh_out = group.createVariable('Vh', 'f', Vh_dim, zlib=True)
        Vh_out[:] = all_Vh[:, :max_k, :]


class EigenDecomposition(Decomposition):
    def __init__(self, variable: Variable):
        self.var = variable

    def decompose(self, output: Dataset, group: Group, var: Variable, levels: np.ma.MaskedArray, dim_species, dim_levels) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(group.name, var.name, var)
        events, _, bound = q.transformed_shape()
        # should be always the same because reshaped variable is square
        all_Q = np.ma.masked_all((events, bound, bound))
        all_s = np.ma.masked_all((events, bound))
        all_k = np.ma.masked_all((events))
        path = f'{group.path}/{var.name}/'
        max_k = 0
        for event in range(var.shape[0]):
            if np.ma.is_masked(levels[event]) or levels.data[event] > 29:
                continue
            level = int(levels.data[event])
            # reduce array dimensions
            matrix = q.transform(var[event][...], level)
            if not self.matrix_ok(event, path, matrix):
                continue
            # test if symmetric
            assert np.allclose(matrix, matrix.T)
            # decompose reduced array
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            if np.iscomplex(eigenvalues).any():
                raise ValueError('Eigenvalues are complex')
            if np.iscomplex(eigenvectors).any():
                raise ValueError('Eigenvectors are complex')
            k = self.target_rank(eigenvalues)
            most_significant = eigenvalues[:k]
            all_k[event] = k
            max_k = max(k, max_k)
            all_Q[event][:eigenvectors.shape[0], :k] = eigenvectors[:, :k]
            all_s[event][:k] = most_significant
        # write all to output
        dimension_name, _ = q.upper_and_lower_dimension()
        target_group = output.createGroup(path)
        target_group.createDimension('rank', size=max_k)
        k_out = target_group.createVariable('k', 'i1', ('event'))
        k_out[:] = all_k[:]
        Q_dim = ('event', dimension_name, 'rank')
        Q_out = target_group.createVariable('Q', 'f', Q_dim, zlib=True)
        Q_out[:] = all_Q[:, :, :max_k]
        s_dim = ('event', 'rank')
        s_out = target_group.createVariable('s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:, :max_k]
