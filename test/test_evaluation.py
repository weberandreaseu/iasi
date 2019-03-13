import luigi
import unittest

from iasi.evaluation import EvaluateCompression


class TestEvaluation(unittest.TestCase):
    def test_evaluate_compression(self):
        task = EvaluateCompression(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force=True,
            # TODO make test not sensitive to leading root '/'
            variable='state/WV/atm_avk'
        )
        assert luigi.build([task], local_scheduler=True)
