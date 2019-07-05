
import logging

import luigi
import numpy as np
import pandas as pd
from cdbw import CDbw
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from iasi.file import FileTask

metrics = {
    'cdbw': CDbw,
    'davis': davies_bouldin_score,
    'sil': silhouette_score,
    'calinski': calinski_harabasz_score
}

logger = logging.getLogger(__name__)

class GridSearch(FileTask):

    params = None
    area = GeographicArea(lat=(-25, 50), lon=(-45, 60))

    def create_pipeline(self):
        raise NotImplementedError

    def output_extension(self):
        return '.csv'

    def statistics(self, y):
        """Statistics about lables

        return: (n_samples, n_cluster, cluster_size_mean, cluster_size_std, n_noise)
        """
        total = len(y)
        cluster, counts = np.unique(y[y > -1], return_counts=True)
        noise = len(y[y == -1])
        return total, len(cluster), counts.mean(), counts.std(), noise

    def load_data(self) -> np.ndarray:
        df = self.area.import_dataset(self.input().path)
        return df.values

    def run(self):
        param_grid = list(ParameterGrid(self.params))
        results = pd.DataFrame(data=param_grid)
        scores = []
        X = self.load_data()
        pipeline = self.create_pipeline()
        for params in param_grid:
            pipeline.set_params(**params)
            scaler = pipeline.named_steps['scaler']
            X_ = scaler.fit_transform(X)
            y = pipeline.fit_predict(X)
            noise_mask = y > -1
            param_score = []
            for name, scorer in metrics.items():
                logger.info(f'Calculate {name} score')
                try:
                    score = scorer(X_[noise_mask], y[noise_mask])
                except ValueError as err:
                    logger.warn(f'Skipped {name}: {err}')
                    score = np.nan
                param_score.append(score)
            scores.append(param_score + list(self.statistics(y)))

        scores = pd.DataFrame(data=scores, columns=list(metrics.keys()) +
                              ['total', 'cluster', 'cluster_mean', 'cluster_std', 'noise'])
        results = pd.concat([results, scores], axis=1)
        with self.output().temporary_path() as file:
            results.to_csv(file)


class GridSearchDBSCAN(GridSearch):

    params = {
        'scaler__km': [60],
        'scaler__H2O': [0.1],
        'scaler__delD': [10],
        'cluster__eps': [2.],
        'cluster__min_samples': [10, 12]
    }

    def output_directory(self):
        return 'dbscan'

    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', SpatialWaterVapourScaler()),
            ('cluster', DBSCAN())
        ])


class GridSearchHDBSCAN(GridSearch):

    params = {
        'scaler__km': [60],
        'scaler__H2O': [0.1],
        'scaler__delD': [10],
        'cluster__min_cluster_size': [10, 12]
    }

    def output_directory(self):
        return 'hdbscan'

    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', SpatialWaterVapourScaler()),
            ('cluster', HDBSCAN())
        ])
