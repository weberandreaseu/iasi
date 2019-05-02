import os
import unittest

import luigi

from iasi.util import CustomTask


class TestCustomTask(unittest.TestCase):
    def test_create_local_target(self):
        task = CustomTask(dst='/tmp/iasi', log=False)
        target = task.create_local_target(
            'custom_target', '28', file='dummy.nc', ext='csv')
        self.assertTrue(os.path.exists('/tmp/iasi/custom_target/28/'))
        self.assertEqual(target.path, '/tmp/iasi/custom_target/28/dummy.csv')
