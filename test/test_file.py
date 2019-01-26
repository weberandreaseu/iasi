import unittest
import luigi
from iasi.file import CopyNetcdfFile


class TestCopyNetcdf(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     assert success

    def test_run_copy(self):
        task = CopyNetcdfFile(
            file='test/resources/IASI-test-single-event.nc',
            dst='/tmp/iasi/copy',
            force=True
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
