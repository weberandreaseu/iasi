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

# file_pattern = 'test/resources/METOPAB_20160101_global_evening_1000.nc'
file_pattern = 'data/input/METOPAB_20160801_global_evening.nc'
area = GeographicArea(lat=(-25, 50), lon=(-45, 60))
df = area.import_dataset(file_pattern)
X = df[features].values

# create estimators
scaler = SpatialWaterVapourScaler()
cluster = DBSCAN(eps=2.6, min_samples=12)
# cluster = HDBSCAN(min_cluster_size=12)

# create pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('cluster', cluster)
])

y = pipeline.fit_predict(X)

subarea = GeographicArea(lat=(-20, 0), lon=(22, 50))
area.subarea_plot(X, y, subarea=subarea, include_noise=True)
