import unittest

import pandas as pd

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler

file = 'test/resources/METOPAB_20160101_global_evening_1000.nc'


class TestData(unittest.TestCase):

    def test_import(self):
        area = GeographicArea(lat=(-25, 50), lon=(-45, 60))
        df = area.import_dataset(file)
        self.assertEqual(df.shape, (541, 4))
        # test filtering
        self.assertGreaterEqual(df.lat.min(), -25)
        self.assertLessEqual(df.lat.max(), 50)
        self.assertGreaterEqual(df.lon.min(), -45)
        self.assertLessEqual(df.lon.max(), 60)


class TestScaler(unittest.TestCase):

    def test_latitude_scaling(self):
        df = pd.DataFrame({
            'lat': [40., 40.],
            'lon': [5., -5.],
            'H2O': [2., 2.],
            'delD': [3., 3.]
        })
        self.assertListEqual(list(df.columns), features)
        scaler = SpatialWaterVapourScaler(delD=10, H2O=0.1, km=50)
        X_ = scaler.fit_transform(df[features].values)

        # lat
        self.assertAlmostEqual(X_[0, 1] * scaler.km, 425, places=0)
        self.assertAlmostEqual(X_[1, 1] * scaler.km, -425, places=0)
        # lon
        self.assertAlmostEqual(X_[0, 0] * scaler.km, 40 * 111)
