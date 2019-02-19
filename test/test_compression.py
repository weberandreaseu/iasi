import unittest

import luigi
import numpy as np
from netCDF4 import Dataset

from iasi import EigenDecomposition, GroupCompression, SingularValueDecomposition


class TestCompression(unittest.TestCase):

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
            vars = nc['/state/WV/atm_avk'].variables.keys()
            self.assertIn('Vh', vars)
            self.assertIn('s', vars)
            self.assertIn('U', vars)

    def test_eigen_decomposition(self):
        task = EigenDecomposition(
            file='test/resources/IASI-test-single-event.nc',
            dst='/tmp/iasi',
            force=True,
            dim=14
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path) as nc:
            vars = nc['/state/WV/atm_avk'].variables.keys()
            self.assertIn('Q', vars)
            self.assertIn('s', vars)

    def test_group_compression(self):
        task = GroupCompression(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path) as nc:
            vars = nc['state/0/WV'].variables.keys()
            self.assertIn('atm', vars)
