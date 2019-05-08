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


eig_winter = import_dataset('data/eigenvalues/METOP*20160201*.nc')
eig_summer = import_dataset('data/eigenvalues/METOP*20160801*.nc')

eig_winter = eig_winter[eig_winter['state/WV/avk/k'] > 0]
eig_summer = eig_summer[eig_summer['state/WV/avk/k'] > 0]


# print(f'Size before filter: {len(filtered)}')
# for column in filtered.columns.tolist()[2:]:
#     print(column)
#     filtered = filtered[filtered[column] >= 0]
# print(f'Size after filter: {len(filtered)}')
# %%


def plot_season_map(summer: pd.DataFrame, winter: pd.DataFrame, var):
    min_k = min(summer[var].min(), winter[var].min())
    max_k = max(summer[var].max(), winter[var].max())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    sc1 = ax1.scatter(
        summer.lon,
        summer.lat,
        c=summer[var],
        marker='.',
        s=2,
        # log scale for error
        vmin=min_k,
        vmax=max_k,
        transform=ccrs.PlateCarree())
    # plt.colorbar(sc1)
    ax1.coastlines()
    ax1.set_title('2016-08-01')
    sc2 = ax2.scatter(
        winter.lon,
        winter.lat,
        c=winter[var],
        marker='.',
        s=2,
        # log scale for error
        vmin=min_k,
        vmax=max_k,
        transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_title('2016-02-01')
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(sc2, cax=cbar_ax)
    plt.savefig('wv_avk_rank_e-3.pdf', bbox_inches='tight')


# %%
plot_season_map(eig_summer, eig_winter, 'state/WV/avk/k')

# %%

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
