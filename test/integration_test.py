import unittest

import luigi
from netCDF4 import Dataset

from iasi import DecompressDataset
from iasi.file import MoveVariables
from test_precision import TestCompareDecompressionResult


class IntegrationTest(TestCompareDecompressionResult):
    @classmethod
    def setUpClass(cls):
        file = '/tmp/data/METOPA_20160625001453_50240_20190209151722.nc'
        compression = DecompressDataset(
            file=file,
            dst='/tmp/iasi',
            force=True
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi',
            force=True
        )
        assert luigi.build([compression, uncompressed], local_scheduler=True)
        cls.compressed = Dataset(compression.output().path)
        cls.uncompressed = Dataset(uncompressed.output().path)
