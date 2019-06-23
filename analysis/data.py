"""
Import spatio-temporal data
"""

import glob
from typing import List, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

Coordinate = Tuple[float, float]
CoordinateRange = Tuple[float, float]

features = ['lat', 'lon', 'H2O', 'delD']


class GeographicArea:
    """Provides methods to import and plot data of a given area"""

    def __init__(self, lat: CoordinateRange = (90, -90), lon: CoordinateRange = (90, -90), level=6):
        """Extend of area in lat lon. Per default all coordinate are included

        :param lat  : Tupel(north, south)
        :param lon  : Tupel(west, east)
        :param level: atmospheric level (0..8)
        """
        self.lat = lat
        self.lon = lon
        self.level = level

    def import_dataset(self, file_pattern: str) -> pd.DataFrame:
        """Import and filter measurements in area matching file pattern"""
        frames = []
        for file in glob.glob(file_pattern):
            frame = pd.DataFrame()
            with Dataset(file) as nc:
                # lat and lon
                for feature in features[:2]:
                    var = nc[feature][...]
                    assert not var.mask.any()
                    frame[feature] = var.data
                # H2O and delD
                for feature in features[2:]:
                    var = nc[feature][:, self.level]
                    assert not var.mask.any()
                    frame[feature] = var.data
            frame = self.filter(frame)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df.lon.between(*self.lon)) &
                  (df.lat.between(self.lat[1], self.lat[0]))]

    def scatter(self, *args, **kwargs):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([*self.lon, *self.lat], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.scatter(*args, **kwargs)
