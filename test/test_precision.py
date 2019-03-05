import unittest
import numpy as np
from iasi import Composition, CompressDataset, MoveVariables, DecompressDataset
from iasi.util import child_variables_of, child_groups_of
from netCDF4 import Dataset, Group, Variable
import luigi
from typing import Set


class TestComposition(unittest.TestCase):
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
        luigi.build([compression, uncompressed], local_scheduler=True)
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

    def test_variable_equality(self):
        for group in child_groups_of(self.uncompressed):
            if group.path == '/':
                other_vars = set(self.compressed.variables.keys())
            else:
                other_vars = set(self.compressed[group.path].variables.keys())
            self.assertSetEqual(set(group.variables.keys()), other_vars)

    def test_reconstruction_of_all_variables(self):
        for group, var in child_variables_of(self.uncompressed['state']):
            pass
            # for each variable try decomposition and composition
            # test if results are nearly equal
