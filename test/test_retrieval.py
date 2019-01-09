import unittest

from iasi import DeltaDRetrieval


class TestDeltaDRetrieval(unittest.TestCase):
    def setUp(self):
        # netCDF file for testing purpose containing a single observation
        self.retrieval = DeltaDRetrieval('data/IASI-event*')

    def test_file_pattern(self):
        description = self.retrieval.describe()
        self.assertEqual(description['n_events'], 1)
        self.assertEqual(description['n_files'], 1)

    def test_retrieve_of_single_event(self):
        df = self.retrieval.retrieve()
        # test shape
        number_columns = len(DeltaDRetrieval.output_variables)
        self.assertEqual(df.shape, (1, number_columns))
        # test column names
        column_names = list(df)
        self.assertListEqual(column_names, DeltaDRetrieval.output_variables)
        # test result of calculated values
        event = df.iloc[0]
        self.assertAlmostEqual(event['H2O'],         1395.876548,   delta=5)
        self.assertAlmostEqual(event['delD'],        -347.37841,    delta=5)
        self.assertAlmostEqual(event['lat'],         70.531105,     delta=5)
        self.assertAlmostEqual(event['lon'],         168.621479,    delta=5)
        self.assertAlmostEqual(event['fqual'],       0.002885,      delta=5)
        self.assertAlmostEqual(event['iter'],        4.0,           delta=5)
        self.assertAlmostEqual(event['srf_flag'],    0,             delta=5)
        self.assertAlmostEqual(event['srf_alt'],     0,             delta=5)
        self.assertAlmostEqual(event['dofs_T2'],     0.923831,      delta=5)
        self.assertAlmostEqual(event['atm_alt'],     4904.0,        delta=5)
        self.assertAlmostEqual(event['Sens'],        0.25728,       delta=5)
        # self.assertAlmostEqual(event['datetime'], )
