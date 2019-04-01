import luigi
import unittest

from iasi.evaluation import EvaluationCompressionSize, EvaluationErrorEstimation


class TestEvaluation(unittest.TestCase):

    def test_compression_size(self):
        task = EvaluationCompressionSize(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force=True,
            # TODO make test not sensitive to leading root '/'
            gases=['WV'],
            variables=['atm_avk']
        )
        assert luigi.build([task], local_scheduler=True)

    def test_error_estimation(self):
        task = EvaluationErrorEstimation(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force=True,
            # TODO make test not sensitive to leading root '/'
            gases=['WV'],
            variables=['atm_avk']
        )
        assert luigi.build([task], local_scheduler=True)
