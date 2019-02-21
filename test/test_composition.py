import logging
import os
import sys
import unittest
from typing import Tuple

import luigi
import numpy as np
from netCDF4 import Dataset, Group, Variable

from iasi import Compositon, GroupCompression, MoveVariables
from iasi.composition import EigenCompositon, SingularValueComposition

# TODO project wide logging configuration
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
# logger.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


class TestComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = GroupCompression(
            file=file,
            dst='/tmp/iasi'
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi'
        )
        luigi.build([compression, uncompressed], local_scheduler=True)
        cls.compressed = Dataset(compression.output().path)
        cls.uncompressed = Dataset(uncompressed.output().path)

    @classmethod
    def tearDownClass(cls):
        cls.compressed.close()
        cls.uncompressed.close()

    def test_singular_value_composition(self):
        pass

    def test_eigen_composition_combined(self):
        self.verify_eigen_composition('/state/WV/atm_n', (1, 2, 2, 29, 29))

    def test_eigen_composition_single(self):
        self.verify_eigen_composition('/state/T/atm_n', (1, 29, 29))

    def verify_eigen_composition(self, attribute: str, shape: Tuple):
        array = self.compressed[attribute]
        nol = self.compressed['atm_nol']
        self.assertIsInstance(array, Group)
        eig = EigenCompositon(array)
        reconstruction = eig.reconstruct(nol)
        self.assertTupleEqual(reconstruction.shape, shape)
        original = self.uncompressed[attribute][...]
        # reconstruction should contain unmasked vales
        self.assertFalse(reconstruction.mask.all())
        # masked values are ignored for comprison
        np.allclose(reconstruction, original)

    @unittest.skip
    def test_group_compression(self):
        task = Compositon(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        # with Dataset(task.output().path) as nc:
        #     vars = nc['/state/WV/atm_avk'].variables.keys()
        #     self.assertIn('Vh', vars)
        #     self.assertIn('s', vars)
        #     self.assertIn('U', vars)
        #     vars = nc['/state/WV/atm_n'].variables.keys()
        #     self.assertIn('Q', vars)
        #     self.assertIn('s', vars)
