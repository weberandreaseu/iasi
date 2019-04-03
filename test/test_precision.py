import logging
import os
import unittest
from typing import Set

import luigi
import numpy as np
from netCDF4 import Dataset, Group, Variable

from iasi import Composition, CompressDataset, DecompressDataset, MoveVariables
from iasi.util import child_groups_of, child_variables_of

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestCompareDecompressionResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
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

    @classmethod
    def tearDownClass(cls):
        cls.compressed.close()
        cls.uncompressed.close()

    def variable_names(self, variables) -> Set:
        return set(map(lambda v: v.name, variables))

    def group_paths(self, groups) -> Set:
        return set(map(lambda g: g.path, groups))

    def test_all_variables_exist(self):
        for group in child_groups_of(self.uncompressed):
            if group.path == '/':
                other_vars = set(self.compressed.variables.keys())
            else:
                other_vars = set(self.compressed[group.path].variables.keys())
            self.assertSetEqual(set(group.variables.keys()), other_vars)

    def test_all_variable_values_are_close(self):
        for group, var in child_variables_of(self.uncompressed['state']):
            path = os.path.join(group.path, var.name)
            original = var[...]
            reconstructed = self.compressed[path][...]
            same_mask = np.equal(original.mask, reconstructed.mask).all()
            self.assertTrue(same_mask, 'reconstruced mask is not equal')
            close = np.ma.allclose(original, reconstructed, atol=1e-3)
            self.assertTrue(close, f'reconstruction values are not close for {path}')
            logger.debug('All variables are close for %s', path)