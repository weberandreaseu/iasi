# %%
"""
Test differen clustering algorithms and score them with different metrics.
Find promising parameter sets using grid search.
"""
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data import GeographicArea, features
from analysis.scaler import SpatialWaterVapourScaler
from analysis.search import GridSearchDBSCAN, GridSearchHDBSCAN

# warnings.filterwarnings("ignore", category=DeprecationWarning)


# file = 'test/resources/METOPAB_20160101_global_evening_1000.nc'
file = 'data/input/METOPAB_20160801_global_evening.nc'
dbscan = GridSearchDBSCAN(file=file, dst='/tmp/cluster')
hdbscan = GridSearchHDBSCAN(file=file, dst='/tmp/cluster')

luigi.build([dbscan, hdbscan], local_scheduler=True)



