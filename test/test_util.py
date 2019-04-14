import os
import unittest

from iasi.util import CustomTask
from iasi import CompressDataset, MoveVariables
import luigi
from netCDF4 import Dataset, Variable, Group
from iasi.util import child_variables_of, child_groups_of
import numpy as np


class TestCustomTask(unittest.TestCase):
    def test_create_local_target(self):
        task = CustomTask(dst='/tmp/iasi', log=False)
        target = task.create_local_target(
            'custom_target', '28', file='dummy.nc', ext='csv')
        self.assertTrue(os.path.exists('/tmp/iasi/custom_target/28/'))
        self.assertEqual(target.path, '/tmp/iasi/custom_target/28/dummy.csv')
