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
            gases=['WV', 'GHG', 'HNO3', 'Tatm'],
            variables=['avk', 'n', 'Tatmxavk'],
            log=False
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
            gases=['WV', 'GHG', 'HNO3', 'Tatm'],
            variables=['avk', 'n', 'Tatmxavk']
        )
        assert luigi.build([task], local_scheduler=True)
        with task.output().open() as file:
            df = pd.read_csv(file)
            cls.wv = df[df['gas'] == 'WV']
            cls.ghg = df[df['gas'] == 'GHG']
            cls.hno3 = df[df['gas'] == 'HNO3']
            cls.atm = df[df['gas'] == 'Tatm']

    def test_water_vapour(self):
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

    def test_greenhouse_gases_report_exists(self):
        self.assertGreater(len(self.ghg), 0)

    def test_nitrid_acid_report_exists(self):
        self.assertGreater(len(self.hno3), 0)

    def test_atmospheric_temperature_report_exists(self):
        self.assertGreater(len(self.hno3), 0)
