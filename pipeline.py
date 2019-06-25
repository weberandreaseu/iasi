# %%
"""
Example of running a clustering pipeline with spatio-temporal data
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from analysis.data import GeographicArea, features
from cdbw import CDbw
from analysis.scaler import SpatialWaterVapourScaler
from analysis.sink import NetCDFSink
from hdbscan import HDBSCAN
import time

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

# file_pattern = 'test/resources/METOPAB_20160101_global_evening_1000.nc'
file_pattern = 'data/input/METOPAB_20160801_global_evening.nc'
area = GeographicArea(lat=(50, -25), lon=(-45, 60))
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
    # 'cluster__min_cluster_size': [4, 5, 8],
    # 'cluster__eps': [3],
    # 'cluster__eps': [3, 5, 7],
    # 'cluster__min_samples': [5, 10, 15]
    'cluster__min_samples': [10]
}))

metrics = {
    # 'cdbw': CDbw,
    'davis': davies_bouldin_score,
    'sil': silhouette_score,
    'calinski': calinski_harabasz_score
}


def statistics(y):
    total = len(y)
    cluster, counts = np.unique(y[y > -1], return_counts=True)
    noise = len(y[y == -1])
    return total, len(cluster), counts.mean(), counts.std(), noise


results = pd.DataFrame(data=param_grid)
scores = []
for i, params in enumerate(param_grid):
    pipeline.set_params(**params)
    scaler = pipeline.named_steps['scaler']
    X_ = scaler.fit_transform(X)
    y = pipeline.fit_predict(X)
    stat = list(statistics(y))
    area.scatter(df['lon'], df['lat'], alpha=1, s=8, c=y)
    plt.show()
    plt.scatter(np.log(X[:, 2]), X[:, 3], alpha=0.15, s=8, c=y)
    plt.show()
    param_score = []
    for scorer in metrics.values():
        try:
            score = scorer(X_, y)
        except ValueError as err:
            score = np.nan
        param_score.append(score)
    # score = [ for scorer in metrics.values()]
    scores.append(param_score + list(statistics(y)))

scores = pd.DataFrame(data=scores, columns=list(metrics.keys()) +
                      ['total', 'cluster', 'cluster_mean', 'cluster_std', 'noise'])
results = pd.concat([results, scores], axis=1)


area.compare_plot(X, y, n_samples=5)