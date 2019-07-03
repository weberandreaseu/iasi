# %%
"""
Test differen clustering algorithms and score them with different metrics.
Find promising parameter sets using grid search.
"""
from random import sample
import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cdbw import CDbw
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from analysis.sink import NetCDFSink

# warnings.filterwarnings("ignore", category=DeprecationWarning)


# file_pattern = 'test/resources/METOPAB_20160101_global_evening_1000.nc'
file_pattern = 'data/input/METOPAB_20160801_global_evening.nc'
area = GeographicArea(lat=(-25, 50), lon=(-45, 60))
df = area.import_dataset(file_pattern)
X = df[features].values

# create estimators
scaler = SpatialWaterVapourScaler()
# cluster = DBSCAN(eps=3, min_samples=5)
cluster = HDBSCAN()

# create pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('cluster', cluster)
])

# create parameter grid
param_grid = list(ParameterGrid({
    'scaler__km': [60],
    'scaler__H2O': [0.1],
    'scaler__delD': [10],
    'cluster__min_cluster_size': [4, 5, 8],
    # 'cluster__eps': [3],
    # 'cluster__eps': [3, 5, 7],
    # 'cluster__min_samples': [5, 10, 15]
    # 'cluster__min_samples': [10]
}))

metrics = {
    # 'cdbw': CDbw,
    'davis': davies_bouldin_score,
    'sil': silhouette_score,
    'calinski': calinski_harabasz_score
}


def statistics(y):
    """Statistics about lables
    
    return: (n_samples, n_cluster, cluster_size_mean, cluster_size_std, n_noise)
    """
    total = len(y)
    cluster, counts = np.unique(y[y > -1], return_counts=True)
    noise = len(y[y == -1])
    return total, len(cluster), counts.mean(), counts.std(), noise


# perform grid search an collect scores and cluster statistics
results = pd.DataFrame(data=param_grid)
scores = []
for i, params in enumerate(param_grid):
    pipeline.set_params(**params)
    scaler = pipeline.named_steps['scaler']
    X_ = scaler.fit_transform(X)
    y = pipeline.fit_predict(X)
    stat = list(statistics(y))
    param_score = []
    for scorer in metrics.values():
        try:
            score = scorer(X_, y)
        except ValueError as err:
            score = np.nan
        param_score.append(score)
    scores.append(param_score + list(statistics(y)))

scores = pd.DataFrame(data=scores, columns=list(metrics.keys()) +
                      ['total', 'cluster', 'cluster_mean', 'cluster_std', 'noise'])
results = pd.concat([results, scores], axis=1)