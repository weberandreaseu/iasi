import unittest
import numpy as np
from iasi import Composition, CompressDataset, MoveVariables
from iasi.util import child_variables_of
from netCDF4 import Dataset, Group, Variable
import luigi


class TestComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = CompressDataset(
            file=file,
            dst='/tmp/iasi',
            force=True
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi',
            force=True
        )
        luigi.build([compression, uncompressed], local_scheduler=True)
        cls.compressed = Dataset(compression.output().path)
        cls.uncompressed = Dataset(uncompressed.output().path)

    @classmethod
    def tearDownClass(cls):
        cls.compressed.close()
        cls.uncompressed.close()

    def test_reconstruction_of_all_variables(self):
        for group, var in child_variables_of(self.uncompressed['state']):
            pass
            # for each variable try decomposition and composition
            # test if results are nearly equal
