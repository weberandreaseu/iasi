import glob
import os

import luigi

from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN, GridSearch
from iasi.file import CustomTask, ReadFile
import pandas as pd
import numpy as np
from typing import Dict


class AggregateClusterStatistics(CustomTask):

    file_pattern = luigi.Parameter()
    clustering_algorithm = luigi.Parameter()
    dst = luigi.Parameter()

    def __init__(self, grid_params, *args, **kwargs):
        self._grid_params = grid_params
        super().__init__(*args, **kwargs)

    @property
    def grid_params(self):
        return self._grid_params

    @grid_params.setter
    def grid_params(self, value):
        self._grid_params = value


    def clustering_task(self) -> GridSearch:
        clustering = {
            'dbscan': GridSearchDBSCAN,
            'hdbscan': GridSearchHDBSCAN
        }
        grid_search = clustering.get(self.clustering_algorithm)
        if not grid_search:
            raise ValueError(
                f'Valid clustering algorithms are {list(clustering.keys())}')
        return grid_search

    def requires(self):
        grid_search = self.clustering_task()
        tasks = []
        for file in glob.glob(self.file_pattern):
            task = grid_search(file=file, dst=self.dst)
            # overwrite default params if set
            if self.grid_params:
                task.grid_params = self.grid_params
            tasks.append(task)
        return tasks

    def output(self):
        path = os.path.join(self.dst, self.clustering_algorithm + '.csv')
        return luigi.LocalTarget(path=path)

    def run(self):
        frames = [pd.read_csv(task.path, index_col=0).reset_index()
                  for task in self.input()]
        df = pd.concat(frames, ignore_index=True)
        def weighted_by_n_cluster(x): return np.average(
            x, weights=df.loc[x.index, 'n_cluster'])
        mapping = {
            # weight cluster scores with total number of measurments
            'davis': {'wm_score': weighted_by_n_cluster},
            'sil': {'wm_score': weighted_by_n_cluster},
            'calinski': {'wm_score': weighted_by_n_cluster},
            'total': ['mean', 'std'],
            'noise': ['mean', 'std'],
            'n_cluster': ['mean', 'std'],
            'cluster_size_mean': {'wm_cluster_size': weighted_by_n_cluster}
        }
        if self.clustering_algorithm == 'hdbscan':
            # DBCV is only defined for HDBSCAN
            mapping['dbcv'] = {'wm_score': weighted_by_n_cluster}

        # get the grid parameter keys of first task.
        # should be same for all tasks
        params = list(next(iter(self.requires())).grid_params.keys())
        # params = list(self.clustering_task().grid_params.keys())
        agg_scores = df.groupby('index').agg(mapping)
        # flatten multiindex columns of agg_scores
        agg_scores.columns = list(map('_'.join, agg_scores.columns.values))
        agg_params = df.groupby('index')[params].mean()
        agg = pd.concat([agg_params, agg_scores], axis=1)

        with self.output().temporary_path() as target:
            agg.to_csv(target, index=None)
