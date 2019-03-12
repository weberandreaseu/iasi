import logging
import os
import sys
import unittest
from typing import Tuple

import warnings


import luigi
import numpy as np
from netCDF4 import Dataset, Group, Variable

from iasi import Composition, CompressDataset, MoveVariables
from iasi.composition import EigenComposition, SingularValueComposition

# TODO project wide logging configuration
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
        assert luigi.build([compression, uncompressed], local_scheduler=True)
        cls.compressed = Dataset(compression.output().path)
        cls.uncompressed = Dataset(uncompressed.output().path)

    @classmethod
    def tearDownClass(cls):
        cls.compressed.close()
        cls.uncompressed.close()

    def masks_equal(self, a, b: np.ma.MaskedArray) -> bool:
        return np.equal(a.mask, b.mask).all()

    def test_eigen_composition_combined(self):
        self.verify_eigen_composition('/state/WV/atm_n', (1, 2, 2, 29, 29))

    def test_eigen_composition_single(self):
        self.verify_eigen_composition('/state/T/atm_n', (1, 29, 29))

    def verify_eigen_composition(self, attribute: str, shape: Tuple):
        array = self.compressed[attribute]
        nol = self.compressed['atm_nol'][...]
        self.assertIsInstance(array, Group)
        eig = EigenComposition(array)
        reconstruction = eig.reconstruct(nol)
        self.assertTupleEqual(reconstruction.shape, shape)
        original = self.uncompressed[attribute][...]
        # reconstruction should contain unmasked vales
        self.assertFalse(reconstruction.mask.all())
        self.assertFalse(np.isnan(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains nans')
        self.assertFalse(np.isinf(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains inf')
        close = np.allclose(reconstruction.compressed(), original.compressed(), atol=1.e-4)
        self.assertTrue(close, 'Eigen composition is not close to original')
        self.assertTrue(self.masks_equal(reconstruction, original))

    def test_svd_one_quadrant(self):
        self.verify_singular_value_composition(
            'state/HNO3/atm_avk', (1, 29, 29))

    def test_svd_two_quadrants(self):
        self.verify_singular_value_composition(
            'state/T/atm2GHGatm_xavk', (1, 2, 29, 29))

    def test_svd_four_quadrants(self):
        self.verify_singular_value_composition(
            'state/WV/atm_avk', (1, 2, 2, 29, 29))

    def verify_singular_value_composition(self, attribute: str, shape: Tuple):
        avk = self.compressed[attribute]
        self.assertIsInstance(avk, Group)
        svc = SingularValueComposition(avk)
        nol = self.compressed['atm_nol'][...]
        reconstruction = svc.reconstruct(nol)
        self.assertTupleEqual(reconstruction.shape, shape)
        original = self.uncompressed[attribute][...]
        # reconstruction should contain unmasked vales
        self.assertFalse(reconstruction.mask.all())
        self.assertFalse(np.isnan(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains nans')
        self.assertFalse(np.isinf(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains inf')
        close = np.allclose(reconstruction.compressed(), original.compressed(), atol=2.e-4)
        self.assertTrue(close, 'Reconstructed data for SVD is not close')
        self.assertTrue(self.masks_equal(reconstruction, original))
