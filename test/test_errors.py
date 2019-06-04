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

    def test_complex_eigenvectors(self):
        """From METOPB_20160112181458_17220_20190313182845.nc 

        eigen decomposition of /state/WV/n:11780 has complex eigenvectors
        """

        task = CompressDataset(
            file='test/resources/MOTIV-complex-eigenvecors.nc',
            force_upstream=True,
            dst='/tmp/iasi'
        )
        assert luigi.build([task], local_scheduler=True)
