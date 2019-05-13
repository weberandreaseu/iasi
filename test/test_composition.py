import logging
import os
import sys
import unittest
import warnings
from typing import Tuple

import luigi
import numpy as np
from netCDF4 import Dataset, Group, Variable

from iasi.composition import (Composition, EigenComposition,
                              SingularValueComposition)
from iasi.compression import CompressDataset
from iasi.file import MoveVariables


class TestComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = CompressDataset(
            file=file,
            dst='/tmp/iasi',
            force=True,
            log=False
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi',
            force=True,
            log=False
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
        self.verify_eigen_composition('/state/WV/n', (1, 2, 2, 29, 29))

    def test_eigen_composition_single(self):
        self.verify_eigen_composition('/state/Tatm/n', (1, 29, 29))

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
        close = np.allclose(reconstruction.compressed(),
                            original.compressed(), atol=1.e-4)
        self.assertTrue(close, 'Eigen composition is not close to original')
        self.assertTrue(self.masks_equal(reconstruction, original))

    def test_svd_one_quadrant(self):
        self.verify_singular_value_composition(
            'state/HNO3/avk', (1, 29, 29))

    def test_svd_two_quadrants(self):
        self.verify_singular_value_composition(
            'state/GHG/Tatmxavk', (1, 2, 29, 29))

    def test_svd_four_quadrants(self):
        self.verify_singular_value_composition(
            'state/WV/avk', (1, 2, 2, 29, 29))

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
        close = np.allclose(reconstruction.compressed(),
                            original.compressed(), atol=2.e-4)
        self.assertTrue(close, 'Reconstructed data for SVD is not close')
        self.assertTrue(self.masks_equal(reconstruction, original))
