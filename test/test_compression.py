import unittest

import luigi
import numpy as np
from netCDF4 import Dataset
from iasi import CompressDataset


class TestCompression(unittest.TestCase):
    
    def test_dataset_compression(self):
        task = CompressDataset(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True,
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)