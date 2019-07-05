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

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN
from analysis.aggregation import AggregateClusterStatistics

warnings.filterwarnings("ignore", category=DeprecationWarning)


# file = 'test/resources/METOPAB_20160101_global_evening_1000.nc'
file = 'data/input/METOPAB_20160801_global_evening.nc'
dbscan = GridSearchDBSCAN(file=file, dst='/tmp/cluster')
# hdbscan = GridSearchHDBSCAN(file=file, dst='/tmp/cluster')

agg_dbscan = AggregateClusterStatistics(
    file_pattern=file,
    dst='data/cluster',
    force_upstream=True,
    clustering_algorithm='dbscan'
)

luigi.build([agg_dbscan], local_scheduler=True)
