import unittest

import luigi
from netCDF4 import Dataset

from iasi.compression import SelectSingleVariable
from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import child_variables_of


class TestCopyNetcdf(unittest.TestCase):

    file = 'test/resources/IASI-test-single-event.nc'

    @unittest.skip('not supported')
    def test_exclude_state(self):
        task = MoveVariables(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi/exclude',
            # any non whitespace string starting with state
            force=True,
            log_file=False
        )
        assert luigi.build([task], local_scheduler=True)
        with Dataset(task.output().path, 'r') as nc:
            with self.assertRaises(IndexError):
                # should not be able to access group 'state'
                nc['state']

    @unittest.skip('created for deprecated task structure')
    def test_copy_full(self):
        task = CopyNetcdfFile(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi/copy',
            force=True,
            log_file=False
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

    @unittest.skip('created for deprecated task structure')
    def test_copy_inclusions(self):
        inclusions = ['state_WVatm', 'state_WVatm_a', 'state_WVatm_avk']
        task = CopyNetcdfFile(
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True,
            exclusion_pattern=r"^((?!state_WVatm+).)*$"
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            vars = list(nc.variables.keys())
            self.assertListEqual(vars, inclusions)

    @unittest.skip('created for deprecated task structure')
    def test_copy_exclusions(self):
        task = CopyNetcdfFile(
            file=self.file,
            dst='/tmp/iasi/copy',
            force=True,
            exclusion_pattern=r"state_WV"
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with Dataset(task.output().path, 'r') as nc:
            vars = list(nc.variables.keys())
            excluded = [
                'state_WVatm', 'state_WVatm_a', 'state_WVatm_avk'
            ]
            for exclusion in excluded:
                self.assertNotIn(exclusion, vars)

    @unittest.skip('created for deprecated task structure')
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

    def test_select_single_variable(self):
        tasks = [SelectSingleVariable(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi/single',
            force_upstream=True,
            variable='state/WV/avk',
            ancestor=ancestor,
            log_file=False
        ) for ancestor in ['MoveVariables', 'CompressDataset', 'DecompressDataset']]
        assert luigi.build(tasks, local_scheduler=True)
        # output from move variables (uncompressed)
        with Dataset(tasks[0].output().path, 'r') as nc:
            child_items = list(child_variables_of(nc))
            self.assertEqual(len(child_items), 1)
            group, var = child_items[0]
            self.assertEqual(group.path, '/state/WV')
            self.assertEqual(var.name, 'avk')
        # output from compress dataset
        with Dataset(tasks[1].output().path, 'r') as nc:
            child_items = list(child_variables_of(nc))
            self.assertEqual(len(child_items), 4)
            vars = [var.name for _, var in child_items]
            self.assertListEqual(vars, ['k', 'U', 's', 'Vh'])
            groups = [group.path for group, _ in child_items]
            self.assertListEqual(groups, ['/state/WV/avk'] * 4)
