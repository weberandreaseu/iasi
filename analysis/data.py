"""
Import spatio-temporal data
"""

import glob
from random import choice, sample
from typing import List, Tuple

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cartopy.crs as ccrs
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

Coordinate = Tuple[float, float]
CoordinateRange = Tuple[float, float]

features = ['lat', 'lon', 'H2O', 'delD']
flags = ['flag_srf', 'flag_cld', 'flag_qual']


class GeographicArea:
    """Provides methods to import and plot data of a given area"""

    def __init__(self, lat: CoordinateRange = (90, -90), lon: CoordinateRange = (90, -90), level=4):
        """Extend of area in lat lon. Per default all coordinate are included

        :param lat  : Tupel(south, north)
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
        return pd.concat(frames, ignore_index=True)[features]

    def filter_location(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df.lon.between(*self.lon)) &
                  (df.lat.between(*self.lat))]

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
        ax.set_extent(self._get_extend(), crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.scatter(*args, **kwargs)
        return ax

    def _get_extend(self):
        return [*self.lon, self.lat[1], self.lat[0]]

    def cluster_subsample(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        sample_frames = []
        cluster = df.groupby(['label']).groups
        for cluster_indices in sample(list(cluster.values()), n_samples):
            sample_frames.append(df.iloc[cluster_indices])
        subsamples = pd.concat(sample_frames)
        return df[df.index.isin(subsamples.index)]

    def cluster_subarea(self, df: pd.DataFrame):
        # calc mean of each cluster
        groups = df.groupby(['label'])['lat', 'lon'].mean()
        # filteder cluser with centroid within coordinate range
        groups = self.filter_location(groups)
        groups['area'] = True
        df = df.merge(groups['area'], left_on='label',
                      right_index=True, how='outer')
        df['area'].fillna(False, inplace=True)
        return df[df['area'] == True]

    def _rectangle(self, **kwargs) -> patches.Rectangle:
        origin = (self.lon[0], self.lat[0])
        width = self.lon[1] - self.lon[0]
        height = self.lat[1] - self.lat[0]
        return patches.Rectangle(
            origin, width, height, **kwargs
        )

    def _set_ticks(self, ax, steps=20):
        # xticks
        start = int(self.lon[0] / 10) * 10
        xticks = np.arange(start, self.lon[1] + steps, steps)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        # yticks
        start = int(self.lat[0] / 10) * 10
        yticks = np.arange(start, self.lat[1] + steps, steps)
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

        # cardinal directions
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)


    def compare_plot(self, X, y,
                     include_noise=True,
                     n_samples=None,
                     subarea=None,
                     filename=None):
        # cosntruct dataframe and filter noise
        df = pd.DataFrame(X, columns=features)
        df['label'] = y
        noise = df[df['label'] == -1]
        no_noise = df[df['label'] > -1]

        # create axes objects
        fig = plt.figure(figsize=(10,4))
        ax1 = plt.subplot(122) # H2O/delD
        ax1.set_xlabel('H2O [log(ppmv)]')
        ax1.set_ylabel('delD [‰]')
        ax2 = plt.subplot(121, projection=ccrs.PlateCarree()) # geo
        ax2.set_extent(self._get_extend(), crs=ccrs.PlateCarree())
        self._set_ticks(ax2)
        ax2.coastlines()

        # plot noise on map
        if include_noise and len(noise) > 0:
            self.geo_scatter(ax2, noise, c='yellow', alpha=0.3)
        
        # plot only n cluster
        if n_samples:
            samples = self.cluster_subsample(df, n_samples)
            self.water_scatter(ax1, samples)
            self.geo_scatter(ax2, samples)

        # plot only cluster in given subarea
        if subarea:
            ax2.add_patch(subarea._rectangle(
                linewidth=1, edgecolor='r', facecolor='none'))
            cluster_area = subarea.cluster_subarea(no_noise)
            outside_area = no_noise[~no_noise.index.isin(cluster_area.index)]
            self.water_scatter(ax1, cluster_area)
            self.geo_scatter(ax2, cluster_area)
            self.geo_scatter(ax2, outside_area)
            # plot only noise in subarea
            noise_area = subarea.filter_location(noise)
            if include_noise and len(noise_area) > 0:
                self.water_scatter(ax1, noise_area, c='yellow', alpha=0.3)
        elif not n_samples:
            if include_noise and len(noise) > 0:
                self.water_scatter(ax1, noise, c='yellow', alpha=0.3)
            self.water_scatter(ax1, no_noise, alpha=0.5)
            self.geo_scatter(ax2, no_noise, alpha=0.5)

        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def geo_scatter(self, ax, df: pd.DataFrame, alpha=0.8, s=8, **kwargs):
        cmap = kwargs.pop('cmap', 'tab20c')
        c = kwargs.pop('c', df['label'])
        ax.scatter(df['lon'], df['lat'], c=c, cmap=cmap,
                   alpha=alpha, s=s, **kwargs)

    def water_scatter(self, ax, df: pd.DataFrame, alpha=1, s=8, **kwargs):
        cmap = kwargs.pop('cmap', 'tab20c')
        c = kwargs.pop('c', df['label'])
        ax.scatter(np.log(df['H2O']), df['delD'], c=c,
                   cmap=cmap,  alpha=alpha, s=s, **kwargs)
