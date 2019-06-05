import datetime
import unittest

import luigi
import numpy as np
from netCDF4 import Dataset

from iasi.compression import (CompressDataset, CompressDateRange,
                              DecompressDataset)


class TestCompression(unittest.TestCase):

    def test_dataset_compression(self):
        task = CompressDataset(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True,
            threshold=0.01,
            log_file=False
        )
        assert luigi.build([task], local_scheduler=True)
        with Dataset(task.output().path) as nc:
            state = nc['state']
            subgroups = state.groups.keys()
            self.assertListEqual(
                list(subgroups), ['GHG', 'HNO3', 'Tatm', 'Tskin', 'WV'])

    def test_dataset_decompression(self):
        task = DecompressDataset(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True,
            log_file=False,
            compress_upstream=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)


class TestDateInterval(unittest.TestCase):

    def test_date_range(self):
        # end date is not inclusive
        interval = luigi.date_interval.Custom.parse('2016-06-01-2016-06-30')
        task = CompressDateRange(date_interval=interval, dst='/tmp/iasi', src='test/resources')
        luigi.build([task], local_scheduler=True)
