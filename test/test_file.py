import unittest

import luigi
from netCDF4 import Dataset

from iasi import CopyNetcdfFile, MoveVariables


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

    def test_copy_exclusions(self):
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

    def test_move_variables(self):
        task = MoveVariables(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi/move',
            force=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            nc: Dataset = nc
            vars = list(nc.variables.keys())
            # there are 39 variables in root group
            self.assertEqual(len(vars), 39)
            state = nc['state']
            compounds = set(state.groups)
            # there are 4 subgroups
            self.assertSetEqual(
                compounds, {'GHG', 'HNO3', 'T', 'WV'})
            for compound in compounds:
                group_variables = set(state[compound].variables.keys())
                expected = {'atm', 'atm_a', 'atm_avk', 'atm_n'}
                self.assertTrue(expected.issubset(group_variables))
