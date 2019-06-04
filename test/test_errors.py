import luigi
import unittest
from iasi import CompressDataset


class TestErrors(unittest.TestCase):
    """Common errors that occur for some events"""

    def test_svd_converges(self):
        task = CompressDataset(
            file='test/resources/MOTIV-svd-did-not-converge.nc',
            force_upstream=True,
            dst='/tmp/iasi'
        )
        assert luigi.build([task], local_scheduler=True)
