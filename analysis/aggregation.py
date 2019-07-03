import glob
import os

import luigi

from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN
from iasi.file import CustomTask, ReadFile


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
