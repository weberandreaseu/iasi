import logging
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset, Group, Variable
from iasi.quadrant import Quadrant
from iasi.util import dimensions_of
import scipy.linalg as la

logger = logging.getLogger(__name__)


class DecompositionException(Exception):
    pass


class Decomposition:

    @classmethod
    def factory(cls, group: Group, variable: Variable, threshold: float = None):
        """If threshold is provided, enforce threshold.
        Else use default values provided by dict above
        """
        # noise matrix (n) is symmetric and is qualified for EigenDecomposition
        if variable.name is 'n' and variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return EigenDecomposition(group, variable, threshold)
        if variable.dimensions[-2:] == ('atmospheric_grid_levels', 'atmospheric_grid_levels'):
            return SingularValueDecomposition(group, variable, threshold)
        raise DecompositionException(
            f'Variable {variable.name} cannot be decomposed')

    def __init__(self, group: Group, variable: Variable, threshold: float = None):
        self.group = group
        self.var = variable
        self.dimensions = dimensions_of(variable)
        self.threshold = threshold if threshold else self.default_threshold(
            group.name, variable.name)

    def default_threshold(self, gas: str, var: str) -> float:
        values = {
            ('WV'  , 'Tatmxavk'): 1e-2,
            ('GHG' , 'Tatmxavk'): 1e-2,
            ('HNO3', 'n')       : 1e-4,
            ('HNO3', 'Tatmxavk'): 1e-2,
            ('Tatm', 'avk')     : 1e-2,
            ('Tatm', 'n')       : 1e-4
        }
        # if nothing else specified, return 1e-3
        return values.get((gas, var), 1e-3)

    def decompose(self, output: Dataset, levels: np.ma.MaskedArray) -> np.ma.MaskedArray:
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
        if np.allclose(matrix, 0, atol=1e-14):
            logger.warning(
                'event %d contains zero or close to zero values in %s. skipping...', event, path)
            ok = False
        return ok
    
    def target_path(self) -> str:
        return f'{self.group.path}/{self.var.name}/'

    def create_target_group(self, output: Dataset) -> Group:
        group = output.createGroup(self.target_path())
        group.eigenvalue_threshold = self.threshold
        return group


class SingularValueDecomposition(Decomposition):

    def decompose(self, output: Dataset, levels: np.ma.MaskedArray) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(self.group.name, self.var.name, self.var)
        events, upper_bound, lower_bound = q.transformed_shape()
        # tranformed shape 1: (e, gl, gl), 2: (e, 2*gl, gl), 4:(e, 2* gl, 2*gl)
        all_U = np.ma.masked_all((events, upper_bound, lower_bound))
        all_s = np.ma.masked_all((events, lower_bound))
        all_Vh = np.ma.masked_all((events, lower_bound, lower_bound))
        all_k = np.ma.masked_all((events), dtype=np.int)
        max_k = 0
        for event in range(self.var.shape[0]):
            if np.ma.is_masked(levels[event]) or levels.data[event] > 29:
                continue
            # reduce array dimensions
            level = int(levels.data[event])
            matrix = q.transform(self.var[event][...], level)
            if not self.matrix_ok(event, self.target_path(), matrix):
                continue
            # decompose reduced array
            try:
                U, s, Vh = la.svd(matrix.data, full_matrices=False, lapack_driver='gesdd')
            except np.linalg.LinAlgError as err:
                logger.error(f'{err} at {self.target_path()}:{event}')
            if np.iscomplex(U).any():
                raise ValueError(
                    f'Left-singuar values are complex for {self.target_path()}:{event}')
            if np.iscomplex(s).any():
                raise ValueError(f'Eigenvalues are complex for {self.target_path()}:{event}')
            if np.iscomplex(Vh).any():
                raise ValueError(
                    f'Right-singlar values are complex for {self.target_path()}:{event}')
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
        group = self.create_target_group(output)
        group.createDimension('rank', size=max_k)
        group.description = f'singular value decomposistion of {self.var.description}. reconstruction with (U * s).dot(Vh)'
        k_out = group.createVariable('k', 'i1', ('event'))
        k_out[:] = all_k[:]
        k_out.description = 'target rank of decomposition (number of eigenvalues)'
        U_dim = ('event', upper_dim, 'rank')
        U_out = group.createVariable('U', 'f', U_dim, zlib=True)
        U_out[:] = all_U[:, :, :max_k]
        U_out = 'left-singular vectors of decompositon'
        s_dim = ('event', 'rank')
        s_out = group.createVariable('s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:, :max_k]
        s_out.description = 'eigenvalues of decomposition'
        Vh_dim = ('event', 'rank', lower_dim)
        Vh_out = group.createVariable('Vh', 'f', Vh_dim, zlib=True)
        Vh_out[:] = all_Vh[:, :max_k, :]
        Vh_out.description = 'transposed right-singular vectors of decompositon'


class EigenDecomposition(Decomposition):

    def decompose(self, output: Dataset, levels: np.ma.MaskedArray) -> np.ma.MaskedArray:
        q: Quadrant = Quadrant.for_assembly(self.group.name, self.var.name, self.var)
        events, _, bound = q.transformed_shape()
        # should be always the same because reshaped variable is square
        all_Q = np.ma.masked_all((events, bound, bound))
        all_s = np.ma.masked_all((events, bound))
        all_k = np.ma.masked_all((events))
        max_k = 0
        for event in range(self.var.shape[0]):
            if np.ma.is_masked(levels[event]) or levels.data[event] > 29:
                continue
            level = int(levels.data[event])
            # reduce array dimensions
            matrix = q.transform(self.var[event][...], level)
            if not self.matrix_ok(event, self.target_path(), matrix):
                continue
            # test if nearlly symmetric
            matrix = matrix.data
            if not np.allclose(matrix, matrix.T):
                raise ValueError(
                    f'Noise matrix is not symmeric for {self.target_path()}:{event}')
            # make matrix symmetric by fixing rounding errors
            matrix = (matrix + matrix.T) / 2
            # decompose matrix
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            # should not be complex anymore because matrix is symmetric (see above)
            if np.iscomplex(eigenvalues).any():
                raise ValueError(f'Eigenvalues are complex for {self.target_path()}:{event}')
            if np.iscomplex(eigenvectors).any():
                raise ValueError(
                    f'Eigenvectors are complex for {self.target_path()}:{event}')
            # covarinace maticies are postive semi definite
            # this implies that eigenvalues are positive
            # unfortuenately, due to floating point errors this is not garantueed
            # to address this problem we assume negative eigenpairs as negligible and filter them
            selected_eigenvalues = []
            selected_eigenvectors = []
            max_eigenvalue = eigenvalues.max()
            for value, vector in zip(eigenvalues, eigenvectors.T):
                if value > 0 and (max_eigenvalue * self.threshold < value):
                    selected_eigenvalues.append(value)
                    selected_eigenvectors.append(vector)
            k = len(selected_eigenvalues)
            max_k = max(k, max_k)
            selected_eigenvectors = np.array(selected_eigenvectors).T
            all_k[event] = k
            all_Q[event][:selected_eigenvectors.shape[0],
                            :k] = selected_eigenvectors[:, :k]
            all_s[event][:k] = selected_eigenvalues
        # write all to output
        dimension_name, _ = q.upper_and_lower_dimension()
        target_group = self.create_target_group(output)
        target_group.createDimension('rank', size=max_k)
        target_group.description = f'eigen decomposition of {self.var.description}. reconstruction with (Q * s).dot(Q.T)'
        k_out = target_group.createVariable('k', 'i1', ('event'))
        k_out[:] = all_k[:]
        k_out.description = 'target rank of decomposition (number of eigenvalues)'
        Q_dim = ('event', dimension_name, 'rank')
        Q_out = target_group.createVariable('Q', 'f', Q_dim, zlib=True)
        Q_out[:] = all_Q[:, :, :max_k]
        Q_out.description = 'eigenvectors of noise matrix'
        s_dim = ('event', 'rank')
        s_out = target_group.createVariable('s', 'f', s_dim, zlib=True)
        s_out[:] = all_s[:, :max_k]
        s_out.description = 'eigenvalues of noise matrix'
