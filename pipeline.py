# %%
"""
Example of running a clustering pipeline with spatio-temporal data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler

file_pattern = 'data/input/METOPAB_20160801_global_evening.nc'
area = GeographicArea(lat=(-25, 50), lon=(-45, 60))
df = area.import_dataset(file_pattern)
X = df[features].values

# create estimators
scaler = SpatialWaterVapourScaler(km=60, H2O=0.1, delD=10)
# cluster = DBSCAN(eps=2.4, min_samples=14)
cluster = HDBSCAN(min_cluster_size=14, gen_min_span_tree=True)

# create pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('cluster', cluster)
])

y = pipeline.fit_predict(X)

subarea = GeographicArea(lat=(-20, 0), lon=(22, 50))
area.subarea_plot(X, y, subarea=subarea, include_noise=True)

# print('dbcv score: ', cluster.relative_validity_)