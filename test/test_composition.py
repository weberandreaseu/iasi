import unittest
import luigi
from iasi import Compositon


class TestComposition(unittest.TestCase):
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
