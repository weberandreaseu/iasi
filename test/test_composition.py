import logging
import os
import sys
import unittest

from netCDF4 import Dataset, Variable, Group

import luigi

from iasi import Compositon, GroupCompression
from iasi.composition import SingularValueComposition, EigenCompositon

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
        # make sure there is a compressed file for testing purpose
        cls.task = GroupCompression(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi'
            # force=True
        )

    def test_compression_output_exists(self):
        path = self.task.output().path
        logger.debug('Compression output path = %s', path)
        self.assertTrue(os.path.isfile(path))

    def test_singular_value_composition(self):
        pass

    def test_eigen_composition(self):
        with Dataset(self.task.output().path) as nc:
            atm_n = nc['/state/WV/atm_n']
            self.assertIsInstance(atm_n, Group)
            eig = EigenCompositon(atm_n)

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
