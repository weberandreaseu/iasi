
import logging

import luigi
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from typing import List

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from iasi.file import FileTask

metrics = {
    'davis': davies_bouldin_score,
    'sil': silhouette_score,
    'calinski': calinski_harabasz_score
}

logger = logging.getLogger(__name__)


class GridSearch(FileTask):

    def __init__(self, grid_params=None, *args, **kwargs):
        self._grid_params = grid_params
        super().__init__(*args, **kwargs)

    area = GeographicArea(lat=(-25, 50), lon=(-45, 60))

    @property
    def grid_params(self):
        if self._grid_params:
            return self._grid_params
        self._grid_params = self.default_parameters()
        return self._grid_params

    @grid_params.setter
    def grid_params(self, value):
        self._grid_params = value

    @classmethod
    def default_parameters(cls):
        raise NotImplementedError

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
        param_grid = list(ParameterGrid(self.grid_params))
        results = pd.DataFrame(data=param_grid)
        scores = []
        X = self.load_data()
        for params in param_grid:
            pipeline = self.create_pipeline()
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
                except Exception as err:
                    logger.warn(f'Skipped {name}: {err}')
                    score = np.nan
                param_score.append(score)
            if isinstance(self, GridSearchHDBSCAN):
                # DBCV score in calaculated by HDBSCAN clusterer
                cluster = pipeline.named_steps['cluster']
                dbcv_score = cluster.relative_validity_
                logger.info(f'Calculate dbcv score {dbcv_score}')
                param_score.append(dbcv_score)
            scores.append(param_score + list(self.statistics(y)))

        if isinstance(self, GridSearchHDBSCAN):
            column_names = list(metrics.keys()) \
                + ['dbcv', 'total', 'n_cluster',
                    'cluster_size_mean', 'cluster_size_std', 'noise']
        else:
            column_names = list(metrics.keys()) \
                + ['total', 'n_cluster',
                   'cluster_size_mean', 'cluster_size_std', 'noise']
        scores = pd.DataFrame(data=scores,  columns=column_names)
        results = pd.concat([results, scores], axis=1)
        with self.output().temporary_path() as file:
            results.to_csv(file)


class GridSearchDBSCAN(GridSearch):

    @classmethod
    def default_parameters(cls):
        return {
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

    @classmethod
    def default_parameters(cls):
        return {
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
            ('cluster', HDBSCAN(gen_min_span_tree=True))
        ])
