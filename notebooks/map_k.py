# %%
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import glob

columns = [
    'lat',
    'lon',
    'state/WV/avk/k',
    'state/WV/n/k',
    'state/WV/Tatmxavk/k',
    'state/GHG/avk/k',
    'state/GHG/n/k',
    'state/GHG/Tatmxavk/k',
    'state/HNO3/avk/k',
    'state/HNO3/n/k',
    'state/HNO3/Tatmxavk/k',
    'state/Tatm/n/k',
    'state/Tatm/avk/k'
]


def import_dataset(file_pattern: str) -> pd.DataFrame:
    frames = []
    for file in glob.glob(file_pattern):
        df = pd.DataFrame()
        with Dataset(file) as nc:
            for column in columns:
                var = nc[column][...]
                # assert not var.mask.any()
                df[column] = np.ma.filled(var)
        frames.append(df)
    # print('Events before filter: %d' % len(df))

    # # filter north/south area
    # df = df[(df.lat < 60) & (df.lat > -15)]
    # # filter west/east area
    # df = df[(df.lon > -60) & (df.lon < 45)]
    # print('Events after filter: %d' % len(df))
    return pd.concat(frames, ignore_index=True)


df = import_dataset('data/eigenvalues/METOP*.nc')
filtered = df
print(f'Size before filter: {len(filtered)}')
for column in filtered.columns.tolist()[2:]:
    print(column)
    filtered = filtered[filtered[column] >= 0]
print(f'Size after filter: {len(filtered)}')

#%%

filtered.drop(['lon', 'lat'], axis=1).boxplot(rot='vertical')
plt.show()

# %% [markdown]
# ### Plots


def plot_eigenvalues_of(variable: str, df=df):
    # filter eigenvalues
    eigenvalues = df[df[variable] >= 0]
    # create map
    ax = plt.axes(projection=ccrs.PlateCarree())
    sc = ax.scatter(eigenvalues.lon,
                    eigenvalues.lat,
                    c=eigenvalues[variable],
                    marker='.',
                    s=2,
                    transform=ccrs.PlateCarree())
    plt.colorbar(sc)
    ax.coastlines()
    plt.show()


# %%
plot_eigenvalues_of('state/WV/avk/k')

# %%
plot_eigenvalues_of('state/WV/n/k')

# %%
plot_eigenvalues_of('state/WV/Tatmxavk/k')

# %%
plot_eigenvalues_of('state/GHG/avk/k')

# %%
plot_eigenvalues_of('state/GHG/n/k')

# %%
plot_eigenvalues_of('state/GHG/Tatmxavk/k')

# %%
plot_eigenvalues_of('state/HNO3/avk/k')

# %%
plot_eigenvalues_of('state/HNO3/n/k')

# %%
plot_eigenvalues_of('state/HNO3/Tatmxavk/k')
