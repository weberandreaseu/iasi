# %%
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd


def import_dataset(file: str) -> pd.DataFrame:
    columns = ['lat', 'lon', 'state/WV/avk/k']
    df = pd.DataFrame()
    with Dataset(file) as nc:
        for column in columns:
            var = nc[column][...]
            # assert not var.mask.any()
            df[column] = np.ma.filled(var)
    # print('Events before filter: %d' % len(df))

    # # filter north/south area
    # df = df[(df.lat < 60) & (df.lat > -15)]
    # # filter west/east area
    # df = df[(df.lon > -60) & (df.lon < 45)]
    # print('Events after filter: %d' % len(df))
    return df


eigenvalues = import_dataset(
    'data/eigenvalues/METOPA_20160201001156_48180_20190323165817.nc')
# %% [markdown]
# ### Filter noise

eigenvalues = eigenvalues[eigenvalues['state/WV/avk/k'] >= 0]

# %%
eigenvalues['state/WV/avk/k'].plot.box()
plt.title('state/WV/avk/k')
plt.show()

# %%
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-65, 50, 65, -20], crs=ccrs.PlateCarree())

sc = ax.scatter(eigenvalues.lon,
                eigenvalues.lat,
                c=eigenvalues['state/WV/avk/k'],
                #    alpha=0.1,
                transform=ccrs.PlateCarree())
plt.colorbar(sc)
ax.coastlines()
plt.show()
