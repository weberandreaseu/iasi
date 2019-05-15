import unittest

import luigi
import pandas as pd

from iasi.aposteriori import DirectAposteriori, SvdAposteriori, EigenAposteriori


class TestAposterioriProcessing(unittest.TestCase):
    def test_uncompressed_retrieval(self):
        task = DirectAposteriori(
            file='test/resources/IASI-test-single-event.nc',
            dst='/tmp/iasi',
            force=True,
            log_file=False
        )
        success = luigi.build([task], local_scheduler=True)
        self.assertTrue(success)
        with task.output().open('r') as file:
            df = pd.read_csv(file)
            self.verify_results(df)


    def verify_results(self, df: pd.DataFrame) -> None:
        # test shape
        # number_columns = len(DeltaDRetrieval.output_variables)
        self.assertEqual(df.shape, (1, 5))
        # test column names
        column_names = list(df)
        self.assertListEqual(column_names, DirectAposteriori.calculated)
        # test result of calculated values
        event = df.iloc[0]
        self.assertAlmostEqual(event['H2O'],         1395.876548,   delta=5)
        self.assertAlmostEqual(event['delD'],        -347.37841,    delta=5)
        # self.assertAlmostEqual(event['lat'],         70.531105,     delta=5)
        # self.assertAlmostEqual(event['lon'],         168.621479,    delta=5)
        # self.assertAlmostEqual(event['fqual'],       0.002885,      delta=5)
        # self.assertAlmostEqual(event['iter'],        4.0,           delta=5)
        # self.assertAlmostEqual(event['srf_flag'],    0,             delta=5)
        # self.assertAlmostEqual(event['srf_alt'],     0,             delta=5)
        self.assertAlmostEqual(event['dofs_T2'],     0.923831,      delta=5)
        self.assertAlmostEqual(event['atm_alt'],     4904.0,        delta=5)
        self.assertAlmostEqual(event['Sens'],        0.25728,       delta=5)
        # self.assertAlmostEqual(event['datetime'], )
