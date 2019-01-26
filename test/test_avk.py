import unittest

import numpy as np
from netCDF4 import Dataset
from iasi import svd


class TestAverageKernel(unittest.TestCase):
    def test_kernel_shape(self):
        with Dataset('test/resources/IASI-test-single-event.nc', 'r') as nc:
            self.assertIsNotNone(nc)
            avk = nc.variables['state_WVatm_avk']
            self.assertEqual(avk.shape, (1, 2, 2, 28, 28))

    def test_singular_value_decomposition(self):
        m, n = 29, 29
        a = np.random.rand(m, n)
        U, s, Vh = svd.decompose(a, reduction_factor=0.8)
        a_ = svd.reconstruct(U, s, Vh)
        self.assertTrue(np.allclose(a, a_, atol=0.1))
