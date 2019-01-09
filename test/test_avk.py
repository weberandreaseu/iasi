import unittest

import numpy as np
from netCDF4 import Dataset
from iasi import svd


class TestAverageKernel(unittest.TestCase):
    def test_kernel_shape(self):
        nc = Dataset(
            'data/IASI-A_20160627_50269_v2018_fast_part0_20180413215350.nc', 'r')
        self.assertIsNotNone(nc)
        avk = nc.variables['state_WVatm_avk']
        self.assertEqual(avk.shape, (12000, 2, 2, 28, 28))
        nc.close()

    def test_singular_value_decomposition(self):
        m, n = 29, 29
        a = np.random.rand(m, n)
        U, s, Vh = svd.decompose(a, reduction_factor=0.8)
        a_ = svd.reconstruct(U, s, Vh)
        self.assertTrue(np.allclose(a, a_, atol=0.1))
