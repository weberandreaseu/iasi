"""
Example of running a clustering pipeline with spatio-temporal data
"""
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from analysis.sink import NetCDFSink

logger = logging.getLogger(__name__)


file_pattern = 'data/input/METOPAB_20160101_global_evening.nc'
area = GeographicArea(lat=(50, -25), lon=(-45, 60))
df = area.import_dataset(file_pattern)
X = df[features].values

# create estimators
scaler = SpatialWaterVapourScaler()
cluster = DBSCAN()

# create pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('cluster', cluster)
])

# create parameter grid
param_grid = ParameterGrid({
    'scaler__km': [60, 80],
    'cluster__eps': [10, 15, 20],
    'cluster__min_samples': [5, 10, 15]
})


format = "{:>10}" * 7
print(format.format('total', 'eps', 'min', 'km', 'noise', 'cluster', 'dbs'))
for params in param_grid:
    pipeline.set_params(**params)
    y = pipeline.fit_predict(X)
    sink = NetCDFSink(pipeline, area)
    sink.satistics(X, y, params)
