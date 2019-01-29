import unittest

import luigi
import numpy as np
from netCDF4 import Dataset

from iasi import SingularValueDecomposition


class TestSingularValueDecomposition(unittest.TestCase):

    def test_svd_conversion(self):
        task = SingularValueDecomposition(
            file='test/resources/IASI-test-single-event.nc',
            dst='/tmp/iasi',
            force=True,
            dim=14
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path) as nc:
            vars = nc.variables.keys()
            self.assertNotIn('state_WVatm_avk', vars)
            self.assertIn('state_WVatm_avk_Vh', vars)
            self.assertIn('state_WVatm_avk_s', vars)
            self.assertIn('state_WVatm_avk_U', vars)


    # @unittest.skip
    # def test_kernel_shape(self):
    #     with Dataset('test/resources/IASI-test-single-event.nc', 'r') as nc:
    #         self.assertIsNotNone(nc)
    #         avk = nc.variables['state_WVatm_avk']
    #         self.assertEqual(avk.shape, (1, 2, 2, 28, 28))

    # @unittest.skip
    # def test_singular_value_decomposition(self):
    #     m, n = 29, 29
    #     a = np.random.rand(m, n)
    #     U, s, Vh = svd.decompose(a, reduction_factor=0.8)
    #     a_ = svd.reconstruct(U, s, Vh)
    #     self.assertTrue(np.allclose(a, a_, atol=0.1))
