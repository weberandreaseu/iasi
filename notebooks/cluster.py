# %%
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    print('Events before filter: %d' % len(df))

    # filter north/south area
    df = df[(df.lat < 60) & (df.lat > -15)]
    # filter west/east area
    df = df[(df.lon < 60) & (df.lon > -45)]
    print('Events after filter: %d' % len(df))
    return df


# %%
df = import_dataset('data/input/METOPAB_20160625_global_evening.nc')

# %%
y = df['delD']
x = df['H2O']
plt.scatter(x, y, alpha=0.1)
plt.ylabel('delD')
plt.xlabel('ln[H2O]')
plt.show()


# %%
plt.scatter(df.lat, df.lon)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
