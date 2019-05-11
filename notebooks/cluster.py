# %%

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
from typing import Tuple, List

Coordinate = Tuple[float, float]
CoordinateRange = Tuple[float, float]


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

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df.lon.between(*self.lon)) & (df.lat.between(self.lat[1], self.lat[0]))]

    def get_extend(self) -> List:
        return [*self.lon, *self.lat]


def import_dataset(file: str) -> pd.DataFrame:
    columns = ['lat', 'lon', 'H2O', 'delD']
    df = pd.DataFrame()
    with Dataset(file) as nc:
        for column in columns:
            var = nc[column][...]
            assert not var.mask.any()
            if column == 'H2O':
                # H2O should be transformed to log scale
                df[column] = np.log(var.data)
            else:
                df[column] = var.data
    return df


# %%
area = GeographicArea(lat=(50, -25), lon=(-45, 60))

# %%
df = import_dataset('data/input/METOPAB_20160625_global_evening.nc')
df = area.filter_dataframe(df)

# %%
y = df['delD']
x = df['H2O']
plt.scatter(x, y, alpha=0.1)
plt.ylabel('delD')
plt.xlabel('ln[H2O]')
plt.show()


# %%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(area.get_extend(), crs=ccrs.PlateCarree())
ax.coastlines()
ax.scatter(df.lon, df.lat, s=4)
plt.show()
