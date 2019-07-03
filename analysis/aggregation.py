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
            raise ValueError(f'Valid clustering algorithms are {list(clustering.keys())}')
        return [grid_search(file=f, dst=self.dst) for f in glob.glob(self.file_pattern)]

    def output(self):
        path = os.path.join(self.dst, self.clustering_algorithm + '.csv')
        return luigi.LocalTarget(path=path)

    def run(self):
        frames = [pd.read_csv(task.path, index_col=0) for task in self.input()]
        df = pd.concat(frames)

        weighted_average = lambda x: np.average(x, weights=df.loc[x.index, 'total'])

        # df.groupby()


        with self.output().temporary_path() as target:
            df.to_csv(target, index=None)