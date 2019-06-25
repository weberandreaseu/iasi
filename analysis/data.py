"""
Import spatio-temporal data
"""

import glob
from typing import List, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

Coordinate = Tuple[float, float]
CoordinateRange = Tuple[float, float]

features = ['lat', 'lon', 'H2O', 'delD']
flags = ['flag_srf', 'flag_cld',
         'flag_qual']


class GeographicArea:
    """Provides methods to import and plot data of a given area"""

    def __init__(self, lat: CoordinateRange = (90, -90), lon: CoordinateRange = (90, -90), level=4):
        """Extend of area in lat lon. Per default all coordinate are included

        :param lat  : Tupel(north, south)
        :param lon  : Tupel(west, east)
        :param level: atmospheric level (0..8). 4 = 4.2 km
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
                for flag in flags:
                    flag_data = nc['/FLAGS/' + flag][...]
                    frame[flag] = flag_data.data
                for flag in ['flag_vres', 'flag_resp']:
                    flag_data = nc['/FLAGS/' + flag][:, self.level]
                    frame[flag] = flag_data.data
            frame = self.filter_location(frame)
            frame = self.filter_flags(frame)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def filter_location(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df.lon.between(*self.lon)) &
                  (df.lat.between(self.lat[1], self.lat[0]))]

    def filter_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df['flag_srf'].isin([0, 1, 5])) &
            (df['flag_cld'].isin([0, 1])) &
            (df['flag_qual'] == 2) &
            (df['flag_vres'] == 2) &
            (df['flag_resp'] == 2)
        ]

    def scatter(self, *args, **kwargs):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([*self.lon, *self.lat], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.scatter(*args, **kwargs)
        return ax

    def compare_plot(self, X, y, include_noise=True, samples=None):
        no_noise = y > -1
        noise = y == -1

        # H2O/delD
        ax1 = plt.subplot(211)
        ax1.scatter(np.log(X[no_noise, 2]), X[no_noise, 3],
                    alpha=1, s=8, c=y[no_noise], cmap='tab20b')
        # geo
        ax2 = plt.subplot(212, projection=ccrs.PlateCarree())
        ax2.set_extent([*self.lon, *self.lat], crs=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.scatter(X[no_noise, 1], X[no_noise, 0],  alpha=1,
                    s=8, c=y[no_noise], cmap='tab20b')

        if include_noise:
            ax1.scatter(np.log(X[noise, 2]), X[noise, 3],
                        alpha=0.5, s=8, c='black')
            ax2.scatter(X[noise, 1], X[noise, 0],  alpha=0.5, s=8, c='black')

        plt.show()
