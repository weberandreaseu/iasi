import unittest

import luigi
from netCDF4 import Dataset

from iasi import CopyNetcdfFile


class TestCopyNetcdf(unittest.TestCase):

    file = 'test/resources/IASI-test-single-event.nc'

    def test_copy_full(self):
        task = CopyNetcdfFile(
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            vars = list(nc.variables.keys())
            expected = [
                'Date', 'Time', 'atm_altitude', 'atm_nol',
                'fit_quality', 'iter', 'lat', 'lon', 'srf_flag',
                'state_WVatm', 'state_WVatm_a', 'state_WVatm_avk'
            ]
            self.assertListEqual(vars, expected)

    def test_copy_inclusions(self):
        inclusions = ['state_WVatm', 'state_WVatm_a', 'state_WVatm_avk']
        task = CopyNetcdfFile(
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True,
            inclusions=inclusions
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            vars = list(nc.variables.keys())
            self.assertListEqual(vars, inclusions)

    def test_copy_exlusions(self):
        exclusions = [
            'Date', 'Time', 'atm_altitude', 'atm_nol',
            'fit_quality', 'iter', 'lat', 'lon', 'srf_flag'
        ]
        task = CopyNetcdfFile(
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True,
            exclusions=exclusions
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            vars = list(nc.variables.keys())
            expected = [
                'state_WVatm', 'state_WVatm_a', 'state_WVatm_avk'
            ]
            self.assertListEqual(vars, expected)

    def test_attribute_error_with_inclusions_and_exclusions(self):
        self.assertRaises(
            AttributeError,
            CopyNetcdfFile,
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True,
            inclusions=['state_WVatm'],
            exclusions=['state_WVatm']
        )
