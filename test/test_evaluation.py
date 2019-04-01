import pandas as pd
import luigi
import unittest

from iasi.evaluation import EvaluationCompressionSize, EvaluationErrorEstimation


class TestEvaluation(unittest.TestCase):

    def test_compression_size(self):
        task = EvaluationCompressionSize(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force_upstream=True,
            gases=['WV'],
            variables=['atm_avk']
        )
        assert luigi.build([task], local_scheduler=True)

    def test_error_estimation(self):
        task = EvaluationErrorEstimation(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force_upstream=True,
            gases=['WV'],
            variables=['atm_avk']
        )
        assert luigi.build([task], local_scheduler=True)
        df = pd.read_csv(task.output()['WV'].path)
        error_wv_16 = df[(df['rc_error'] == False) & (
            df['level_of_interest'] == -16)]
        self.assertAlmostEqual(error_wv_16.err.values[0], 0.12626536984680686)
        rc_error_wv_16 = df[(df['rc_error']) & (
            df['level_of_interest'] == -16) & (df['threshold'] == 0.001)]
        self.assertAlmostEqual(rc_error_wv_16.err.values[0], 1.510003e-08)
