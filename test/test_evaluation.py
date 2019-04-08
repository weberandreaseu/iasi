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
            force=True,
            gases=['WV', 'GHG', 'HNO3'],
            variables=['avk', 'n', 'Tatmxavk']
        )
        assert luigi.build([task], local_scheduler=True)

    @classmethod
    def filter_by(cls, df, var: str, level_of_interest: int, type: int = 1, rc_error: bool = True, threshold: float = None) -> pd.DataFrame:
        filtered = df[
            (df['var'] == var) &
            (df['level_of_interest'] == level_of_interest) &
            (df['type'] == type) &
            (df['rc_error'] == rc_error)
        ]
        if threshold:
            filtered = filtered[filtered['threshold'] == threshold]
        assert len(filtered) > 0
        return filtered

    @classmethod
    def setUpClass(cls):
        task = EvaluationErrorEstimation(
            file='test/resources/MOTIV-single-event.nc',
            # file='data/input/MOTIV-slice-1000.nc',
            dst='/tmp/iasi',
            force_upstream=True,
            gases=['WV', 'GHG', 'HNO3'],
            variables=['avk', 'n', 'Tatmxavk']
        )
        assert luigi.build([task], local_scheduler=True)
        cls.wv = pd.read_csv(task.output()['WV'].path)
        cls.ghg = pd.read_csv(task.output()['GHG'].path)
        cls.hno3 = pd.read_csv(task.output()['HNO3'].path)

    def verify_water_vapour(self):
        ##### type 1 error #####

        # water vapour: level 16
        # error
        err_wv_avk_type1 = self.filter_by(
            self.wv, 'avk', -16, rc_error=False
        )
        self.assertEqual(len(err_wv_avk_type1), 1,
                         'More results than expected')
        self.assertAlmostEqual(
            err_wv_avk_type1.err.values[0],
            0.0786345470381124,
            msg='Wrong value for type 1 error'
        )

        # reconstruction error
        rc_err_wv_avk_type1 = self.filter_by(
            self.wv, 'avk', -16, rc_error=True, threshold=0.001
        )
        self.assertEqual(len(rc_err_wv_avk_type1), 1,
                         'More results than expected')
        self.assertAlmostEqual(
            rc_err_wv_avk_type1.err.values[0],
            1.510003e-08,
            msg='Wrong value for type 1 rc_error'
        )

        ##### type 2 error #####
        err_wv_avk_type2 = self.filter_by(
            self.wv, 'avk', -16, rc_error=False, type=2
        )
        self.assertEqual(len(err_wv_avk_type2), 1,
                         'More results than expected')
        rc_err_wv_avk_type2 = self.filter_by(
            self.wv, 'avk', -16, rc_error=True, type=2, threshold=0.001
        )
        self.assertEqual(len(rc_err_wv_avk_type2), 1,
                         'More results than expected')
        self.assertLess(
            rc_err_wv_avk_type1.err.values[0],
            rc_err_wv_avk_type2.err.values[0],
            'Type 1 rc_error should be smaller than type 2 rc_error'
        )
        self.assertLess(
            err_wv_avk_type1.err.values[0],
            err_wv_avk_type2.err.values[0],
            'Type 1 error should be smaller than type 2 error'
        )

    def verify_greenhouse_gases(self, ghg: pd.DataFrame):
        self.assertGreater(len(self.ghg), 0)

    def verify_nitrid_acid(self, ghg: pd.DataFrame):
        self.assertGreater(len(self.hno3), 0)
