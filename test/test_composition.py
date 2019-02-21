import logging
import os
import sys
import unittest
from typing import Tuple

import warnings


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
logger.setLevel(logging.DEBUG)


class TestComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = GroupCompression(
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
        self.assertFalse(np.isnan(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains nans')
        self.assertFalse(np.isinf(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains inf')
        np.allclose(reconstruction.data, original.data)

    def test_svd_one_quadrant(self):
        self.verify_singular_value_composition(
            'state/HNO3/atm_avk', (1, 29, 29))

    def test_svd_two_quadrants(self):
        self.verify_singular_value_composition(
            'state/T/atm2GHGatm_xavk', (1, 2, 29, 29))

    def test_svd_four_quadrants(self):
        self.verify_singular_value_composition(
            'state/WV/atm_avk', (1, 2, 2, 29, 29))

    def verify_singular_value_composition(self, arrtribute: str, shape: Tuple):
        avk = self.compressed[arrtribute]
        self.assertIsInstance(avk, Group)
        svc = SingularValueComposition(avk)
        reconstruction = svc.reconstruct(None)
        self.assertTupleEqual(reconstruction.shape, shape)
        original = self.uncompressed['state/WV/atm_avk'][...]
        # reconstruction should contain unmasked vales
        self.assertFalse(reconstruction.mask.all())
        self.assertFalse(np.isnan(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains nans')
        self.assertFalse(np.isinf(reconstruction[:, :28, :28]).any(
        ), 'Reconstructed array contains inf')
        np.allclose(reconstruction.data, original.data)
