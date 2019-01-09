from typing import Tuple

import numpy as np
from scipy import linalg
from math import ceil

import logging

def decompose(matrix: np.ndarray, reduction_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.debug(f'Decompose matrix of shape {matrix.shape}')
    U, s, Vh = linalg.svd(matrix, full_matrices=False)
    logging.debug(
        f'Original shape of SVD U={U.shape}, s={s.shape}, Vh={Vh.shape}')

    dim = ceil(s.shape[0] * reduction_factor)
    logging.debug(f'Dim size to {dim}')
    s = s[:dim]
    U = U[:, :dim]
    Vh = Vh[:dim, :]

    logging.debug(
        f'Reduced shape of SVD U={U.shape}, s={s.shape}, Vh={Vh.shape}')
    svd_size = U.size + s.size + Vh.size
    logging.info(
        f'Matrix size:{matrix.size} , SVD size: {svd_size}, Relative size: {100 * svd_size / matrix.size:.2f}%')
    return (U, s, Vh)


def reconstruct(U: np.ndarray, s: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    sigma = np.diag(s)
    return np.dot(U, np.dot(sigma, Vh))
