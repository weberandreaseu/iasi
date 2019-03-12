import luigi
import unittest

from iasi.evaluation import EvaluateCompression


class TestEvaluation(unittest.TestCase):
    def test_evaluate_compression(self):
        task = EvaluateCompression(
            file='test/resources/MOTIV-single-event.nc',
            dst='/tmp/iasi',
            force=True
        )
        assert luigi.build([task], local_scheduler=True)
