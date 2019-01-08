from typing import Tuple

import numpy as np
from scipy import linalg


def decompose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print('Decompose matrix of shape {}'.format(matrix.shape))
    U, s, Vh = linalg.svd(matrix)
    print('Shape U={}, s={}, Vh={}'.format(U.shape, s.shape, Vh.shape))
    return (U, s, Vh)


def reconstruct(U: np.ndarray, s: np.ndarray, Vh: np.ndarray, reduction_factor=0.5) -> np.ndarray:
    shape = (U.shape[0], Vh.shape[0])
    sigma = np.zeros(shape)
    for i in range(min(shape)):
        sigma[i, i] = s[i]
    return np.dot(U, np.dot(sigma, Vh))
