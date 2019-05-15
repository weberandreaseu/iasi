import unittest

import luigi
from netCDF4 import Dataset

from iasi.compression import SelectSingleVariable
from iasi.file import CopyNetcdfFile, MoveVariables
from iasi.util import child_variables_of


class TestCopyNetcdf(unittest.TestCase):

    file = 'test/resources/MOTIV-single-event.nc'

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
