import glob
import os

import luigi

from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN, GridSearch
from iasi.file import CustomTask, ReadFile
import pandas as pd
import numpy as np


class AggregateClusterStatistics(CustomTask):

    file_pattern = luigi.Parameter()
    clustering_algorithm = luigi.Parameter()
    dst = luigi.Parameter()

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
        return [grid_search(file=f, dst=self.dst) for f in glob.glob(self.file_pattern)]

    def output(self):
        path = os.path.join(self.dst, self.clustering_algorithm + '.csv')
        return luigi.LocalTarget(path=path)

    def run(self):
        frames = [pd.read_csv(task.path, index_col=0).reset_index()
                  for task in self.input()]
        df = pd.concat(frames, ignore_index=True)
        def weighted_mean_score(x): return np.average(
            x, weights=df.loc[x.index, 'total'])
        def weighted_mean_cluster_size(x): return np.average(
            x, weights=df.loc[x.index, 'cluster'])
        mapping = {
            # weight cluster scores with total number of measurments
            'davis': {'wm_score': weighted_mean_score},
            'sil': {'wm_score': weighted_mean_score},
            'calinski': {'wm_score': weighted_mean_score},
            'total': ['mean', 'std'],
            'noise': ['mean', 'std'],
            'cluster': ['mean', 'std'],
            'cluster_mean': {'wm_cluster_size': weighted_mean_cluster_size}
        }
        params = list(self.clustering_task().params.keys())
        agg_scores = df.groupby('index').agg(mapping)
        # flatten multiindex columns of agg_scores
        agg_scores.columns = list(map('_'.join, agg_scores.columns.values))
        agg_params = df.groupby('index')[params].mean()
        agg = pd.concat([agg_params, agg_scores], axis=1)

        with self.output().temporary_path() as target:
            agg.to_csv(target, index=None)
