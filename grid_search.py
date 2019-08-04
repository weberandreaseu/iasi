# %%
"""
Test differen clustering algorithms and score them with different metrics.
Find promising parameter sets using grid search.
"""
import warnings

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.aggregation import AggregateClusterStatistics
from analysis.data import GeographicArea
from analysis.scaler import SpatialWaterVapourScaler
from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN

warnings.filterwarnings("ignore", category=DeprecationWarning)

file_pattern = 'test/resources/METOPAB_*global_evening*.nc'

dbscan_params = {
    'scaler__km': [50, 60, 70, 80],
    'scaler__H2O': [0.11, 0.12, 0.13, 0.14],
    'scaler__delD': [6, 7, 8, 10],
    'cluster__eps': [1.8, 2, 2.2, 2.4],
    'cluster__min_samples': [8, 10, 12, 14],
}

hdbscan_params = {
    'scaler__km': [50, 60, 70, 80],
    'scaler__H2O': [0.11, 0.12, 0.13, 0.14],
    'scaler__delD': [6, 7, 8, 10],
    'cluster__min_cluster_size': [10, 12, 14]
}

agg_dbscan = AggregateClusterStatistics(
    dbscan_params,
    file_pattern=file_pattern,
    dst='/tmp/cluster/grid',
    force_upstream=True,
    clustering_algorithm='dbscan',
)

agg_hdbscan = AggregateClusterStatistics(
    hdbscan_params,
    file_pattern=file_pattern,
    dst='/tmp/cluster/grid',
    force_upstream=True,
    clustering_algorithm='hdbscan',
)

luigi.build([agg_dbscan, agg_hdbscan], local_scheduler=True)
