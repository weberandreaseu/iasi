# %%

import glob
from typing import List, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Coordinate = Tuple[float, float]
CoordinateRange = Tuple[float, float]

features = ['lat', 'lon', 'H2O', 'delD']


class GeographicArea:
    """
    """

    def __init__(self, lat: CoordinateRange, lon: CoordinateRange):
        """Extend of area in lat lon

        :param lat  : Tupel(north, south)
        :param lon  : Tupel(west, east)
        """
        self.lat = lat
        self.lon = lon

    def import_dataset(self, file_pattern: str) -> pd.DataFrame:
        frames = []
        for file in glob.glob(file_pattern):
            frame = pd.DataFrame()
            with Dataset(file) as nc:
                for feature in features:
                    var = nc[feature][...]
                    assert not var.mask.any()
                    frame[feature] = var.data
            # filter to given area
            frame = frame[(frame.lon.between(*self.lon)) &
                          (frame.lat.between(self.lat[1], self.lat[0]))]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def get_extend(self) -> List:
        return [*self.lon, *self.lat]


class SpatialWaterVapourScaler(BaseEstimator, TransformerMixin):
    """Scale water vapour features including latitude and longitude:
    - 1 degree lat = 111.00 km
    - 1 degree lon = 110.57 km (at equator)
    source: https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616
    """

    def __init__(self, delD=10, H2O=0.1, km=60):
        self.delD = delD
        self.H2O = H2O
        self.km = km

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Scale 4 water vapour features according to transform parameters

        order of colums:
        - lat
        - lon
        - H2O
        - delD
        """
        assert X.shape[1] == 4
        # lat
        X[:, 0] = (X[:, 0] * 111) / self.km
        # lon
        X[:, 1] = (X[:, 1] * 110.57) / self.km
        # H2O
        X[:, 2] = np.log(X[:, 2]) / self.H2O
        # delD
        X[:, 3] = X[:, 3] / self.delD
        return X


# %%
area = GeographicArea(lat=(50, -25), lon=(-45, 60))
df = area.import_dataset('data/input/METOPAB_20160625_global_evening.nc')
scaler = SpatialWaterVapourScaler(km=120, H2O=0.2, delD=5)
# scaled = pd.DataFrame(scaler.transform(df.values))
clustering = DBSCAN(eps=1.5, min_samples=10)
pipeline = Pipeline([
    ('scaler', scaler),
    ('clustering', clustering)
])
samples = df
pipeline.fit(samples[features].values)
clustering = pipeline.named_steps['clustering']
samples['label'] = clustering.labels_


# %%
y = samples['delD']
x = np.log(samples['H2O'])
plt.scatter(x, y, alpha=0.15, marker='.', s=8, c=samples['label'])
plt.ylabel('delD')
plt.xlabel('ln[H2O]') 
plt.show()


# %%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(area.get_extend(), crs=ccrs.PlateCarree())
ax.coastlines()
ax.scatter(samples.lon, samples.lat, alpha=0.65, marker='.', s=8, c=samples['label'])
plt.show()
