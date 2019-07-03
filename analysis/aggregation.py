import glob
import os

import luigi

from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN
from iasi.file import CustomTask, ReadFile
import pandas as pd
import numpy as np


class AggregateClusterStatistics(CustomTask):

    file_pattern = luigi.Parameter()
    clustering_algorithm = luigi.Parameter()
    dst = luigi.Parameter()

    def requires(self):
        clustering = {
            'dbscan': GridSearchDBSCAN,
            'hdbscan': GridSearchHDBSCAN
        }
        grid_search = clustering.get(self.clustering_algorithm)
        if not grid_search:
            raise ValueError(
                f'Valid clustering algorithms are {list(clustering.keys())}')
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
        df = df.groupby(['index']).agg(mapping)

        with self.output().temporary_path() as target:
            df.to_csv(target, index=None)
