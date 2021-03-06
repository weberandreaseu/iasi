import unittest

import pandas as pd

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from analysis.search import GridSearchHDBSCAN, GridSearchDBSCAN
from analysis.aggregation import AggregateClusterStatistics
from sklearn.model_selection import ParameterGrid
import luigi

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


class TestGridSearch(unittest.TestCase):

    def test_hdbscan(self):
        task = GridSearchHDBSCAN(
            file=file,
            dst='/tmp/cluster',
            force_upstream=True
        )
        assert luigi.build([task], local_scheduler=True)
        df = pd.read_csv(task.output().path)
        columns = set(df.columns)
        expected = {'total', 'n_cluster', 'cluster_size_mean',
                    'cluster_size_std', 'noise'}
        self.assertTrue(expected <= columns)
        self.assertEqual(df.shape[1], 14)

    def test_dbscan(self):
        task = GridSearchDBSCAN(
            file=file,
            dst='/tmp/cluster',
            force_upstream=True
        )
        assert luigi.build([task], local_scheduler=True)
        df = pd.read_csv(task.output().path)
        columns = set(df.columns)
        expected = {'total', 'n_cluster', 'cluster_size_mean',
                    'cluster_size_std', 'noise'}
        self.assertTrue(expected <= columns)
        self.assertEqual(df.shape, (2, 14))

    def test_aggregation(self):
        grid_params = {
            'scaler__km': [60],
            'scaler__H2O': [0.1],
            'scaler__delD': [10],
            'cluster__eps': [2.],
            'cluster__min_samples': [10, 12]
        }
        task = AggregateClusterStatistics(
            grid_params,
            file_pattern='test/resources/METOPAB*evening_1000.nc',
            dst='/tmp/cluster',
            clustering_algorithm='dbscan',
            force_upstream=True
        )
        assert luigi.build([task], local_scheduler=True)
        with task.output().open() as out:
            df = pd.read_csv(out)
            self.assertEqual(df.shape, (2, 15))
