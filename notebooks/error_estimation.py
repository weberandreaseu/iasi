# %% [markdown]
# # Error Estimation
# %%
from datetime import datetime
import glob
import logging
import os

import cartopy.crs as ccrs
import luigi
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 0]

# %% [markdown]

# ## Data import
# Import functions to read error and size from csv files


def altitude_by(level_of_interest: int):
    map = {
        -6: '22.1 km',
        -19: '4.2 km',
        -16: '6.3 km',
        -10: '11.9 km',
        -35: '22.1 km',
        -39: '11.9 km',
        -48: '4.2 km',
        0: '0 km'
    }
    return map.get(level_of_interest)


def import_data(path_pattern: str,
                gas: str = None,
                var: str = None,
                threshold: float = None,
                loi: int = None,
                type: int = None,
                inlcude_coordinates=False) -> pd.DataFrame:
    frames = []
    for file in glob.glob(path_pattern):
        frame = pd.read_csv(file, index_col=None, header=0)
        # filter to gas and variables
        if loi:
            frame = frame[frame['level_of_interest'] == loi]
        if gas:
            gas = gas.split('_')[0]
            frame = frame[frame['gas'] == gas]
        if var:
            frame = frame[frame['var'] == var]
        # set threshold == 0 to 1 for consistency
        if type:
            frame = frame[frame['type'] == type]
        frame['threshold'].replace(0, 1, inplace=True)
        if threshold:
            frame = frame[frame['threshold'] == threshold]
        if 'level_of_interest' in frame:
            # set altitudes
            frame['altitude'] = frame['level_of_interest'].apply(altitude_by)
        if 'size' in frame:
            frame['size'] = frame['size'] / 1000
        else:
            # distinguish between N2O (nol > -29) and CH4 (nol < -29)
            frame.loc[(frame['gas'] == 'GHG') & (
                frame['level_of_interest'] >= -29), 'gas'] = 'GHG_N2O'
            frame.loc[(frame['gas'] == 'GHG') & (
                frame['level_of_interest'] < -29), 'gas'] = 'GHG_CH4'
        if inlcude_coordinates:
            _, filename = os.path.split(file)
            orbit, _ = os.path.splitext(filename)
            nc = Dataset(f'data/eigenvalues/{orbit}.nc')
            lat = nc['lat'][...]
            lon = nc['lon'][...]
            coordinates = np.block([[lat], [lon]])
            coordinates = pd.DataFrame(coordinates.T, columns=['lat', 'lon'])
            frame = frame.merge(coordinates, left_on='event', right_index=True)
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True)


def import_size(path_pattern: str):
    frames = []
    for file in glob.glob(path_pattern):
        # frame = pd.read_csv(file, index_col=None, header=0)
        frame = pd.read_csv(file)
        if 'size' in frame:
            frame['size'] = frame['size'] / 1000
        datestring = file.split('_')[1]
        date = datetime.strptime(datestring[:8], '%Y%m%d')
        frame['date'] = date.date()
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True)


# %% [markdown]
# ## Plot functions

# Plot error and size for different seasons, threholds products etc.

def plot_error_map(df, gas=None, var=None, season=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    sc = ax.scatter(
        df.lon,
        df.lat,
        c=df['err'],
        marker='.',
        s=2,
        # log scale for error
        norm=colors.LogNorm(df.err.min(), df.err.max()),
        transform=ccrs.PlateCarree())
    plt.colorbar(sc)
    ax.coastlines()
    if season:
        plt.savefig(f'map_{gas}_{var}_{season}.pdf')
    plt.show()


def plot_season_map(summer: pd.DataFrame, winter: pd.DataFrame):
    min_err = min(summer.err.min(), winter.err.min())
    max_err = max(summer.err.max(), winter.err.max())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    sc1 = ax1.scatter(
        summer.lon,
        summer.lat,
        c=summer['err'],
        marker='.',
        s=1,
        # log scale for error
        norm=colors.LogNorm(min_err, max_err),
        vmin=min_err,
        vmax=max_err,
        transform=ccrs.PlateCarree())
    # plt.colorbar(sc1)
    ax1.coastlines()
    ax1.set_title('2016-08-01')
    sc2 = ax2.scatter(
        winter.lon,
        winter.lat,
        c=winter['err'],
        marker='.',
        s=1,
        # log scale for error
        norm=colors.LogNorm(min_err, max_err),
        vmin=min_err,
        vmax=max_err,
        transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_title('2016-02-01')
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(sc2, cax=cbar_ax)
    plt.savefig('avk_winter_summer.pdf')


def plot_size_season(df: pd.DataFrame, gas: str, var: str):
    original = df[
        (df['ancestor'] == 'MoveVariables') &
        (df['gas'] == gas) &
        (df['variable'] == var)
    ]
    compressed = df[
        (df['ancestor'] == 'CompressDataset') &
        (df['gas'] == gas) &
        (df['variable'] == var)
    ]
    ax = compressed.groupby(['threshold', 'date']).mean()[
        'size'].unstack().plot.bar(rot=0, legend=True, figsize=(4, 4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, (name, group) in enumerate(original.groupby(['date'])):
        group_mean = group.mean()['size']
        ax.axhline(group_mean, color=colors[index], linestyle='--')
    ax.invert_xaxis()
    ax.set_ylabel('mean orbit size [MB]')
    ax.set_xlabel('threshold eigenvalue selection')
    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    plt.title(f'Data size reduction for {gas} {var}')
    plt.show()


def plot_levels(df: pd.DataFrame, gas: str, var: str, type, season=None):
    df = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['type'] == type)
    ]

    # plot reconstruction error
    original = df[df['threshold'] == 1]
    compressed = df[~df.isin(original).all(1)]
    groups = compressed.groupby(['threshold', 'altitude']).mean()['err']
    print(groups)
    ax = groups.unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    ax.invert_xaxis()

    # plot original error
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, (name, group) in enumerate(original.groupby(['altitude'])):
        group_mean = group.mean()['err']
        ax.axhline(group_mean, color=colors[index], linestyle='--')

    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    # plt.title(f'Error estimation for {gas} {var} type {type}')
    ax.set_xlabel('threshold for eigenvalue selection')
    ax.set_ylabel('error variance [log(ppmv)^2]')
    plt.savefig(f'err_{gas}_{var}_type{type}.pdf')
    plt.show()


def plot_types(df: pd.DataFrame, gas: str, var: str, level_of_interest: int):
    df = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['level_of_interest'] == level_of_interest)
    ]
    original = df[df['rc_error'] == False]
    compressed = df[~df.isin(original).all(1)]

    ax = compressed.groupby(['threshold', 'type']).mean()[
        'err'].unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    # plt.title(
    #     f'Error estimation {gas} {var} with level of interest {level_of_interest}')
    ax.invert_xaxis()
    ax.set_xlabel('threshold eigenvalue selection')
    ax.set_ylabel('error variance [log(ppmv)^2]')

    # plot original error
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, (name, group) in enumerate(original.groupby(['type'])):
        group_mean = group.mean()['err']
        ax.axhline(group_mean, color=colors[index], linestyle='--')

    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    plt.savefig(f'err_{gas}_{var}_loi{level_of_interest}.pdf')
    plt.show()


# %% [markdown]

# ## Example: Water Vapour Averaging Kernel
# plot size
size = import_size('data/final/size/METOP*.csv')
plot_size_season(size, 'WV', 'avk')

# %%
err = import_data('data/final/WV/avk/METOPB_20160801115357_20086_20190312071833.nc',
                  inlcude_coordinates=True)
plot_levels(err, 'WV', 'avk', type=1)

# %%
plot_types(err, 'WV', 'avk', -19)

# %%
# error map
err_winter = import_data('data/final/WV/avk/METOPA_20160201*.nc',
                         gas='WV',
                         var='avk',
                         inlcude_coordinates=True,
                         loi=-19,
                         threshold=0.001,
                         type=1)
err_summer = import_data('data/final/WV/avk/METOP*_20160801*.nc',
                         gas='WV',
                         var='avk',
                         inlcude_coordinates=True,
                         loi=-19,
                         threshold=0.001,
                         type=1)

# %%
plot_error_map(err_winter, 'WV', 'avk')

# %%
plot_season_map(err_summer, err_winter)

# %%
# Batch processing: define all plot needed
plots = [
    # WV avk
    {'method': plot_levels, 'gas': 'WV', 'var': 'avk', 'type': 1},
    {'method': plot_levels, 'gas': 'WV', 'var': 'avk', 'type': 2},
    {'method': plot_types, 'gas': 'WV', 'var': 'avk', 'level_of_interest': -19},
    {'method': plot_types, 'gas': 'WV', 'var': 'avk', 'level_of_interest': -16},
    # WV n
    {'method': plot_levels, 'gas': 'WV', 'var': 'n', 'type': 1},
    {'method': plot_levels, 'gas': 'WV', 'var': 'n', 'type': 2},
    {'method': plot_types, 'gas': 'WV', 'var': 'n', 'level_of_interest': -19},
    {'method': plot_types, 'gas': 'WV', 'var': 'n', 'level_of_interest': -16},
    # WV Tatmxavk
    {'method': plot_levels, 'gas': 'WV', 'var': 'Tatmxavk', 'type': 1},
    {'method': plot_levels, 'gas': 'WV', 'var': 'Tatmxavk', 'type': 2},
    {'method': plot_types, 'gas': 'WV', 'var': 'Tatmxavk', 'level_of_interest': -19},
    {'method': plot_types, 'gas': 'WV', 'var': 'Tatmxavk', 'level_of_interest': -16},
    # GHG
    {'method': plot_levels, 'gas': 'GHG_CH4', 'var': 'avk', 'type': 1},
    {'method': plot_levels, 'gas': 'GHG_N2O', 'var': 'avk', 'type': 1},
    {'method': plot_levels, 'gas': 'GHG_CH4', 'var': 'n', 'type': 1},
    {'method': plot_levels, 'gas': 'GHG_N2O', 'var': 'n', 'type': 1},
    {'method': plot_levels, 'gas': 'GHG_CH4', 'var': 'Tatmxavk', 'type': 1},
    {'method': plot_levels, 'gas': 'GHG_N2O', 'var': 'Tatmxavk', 'type': 1},
    # HNO3
    {'method': plot_levels, 'gas': 'HNO3', 'var': 'avk', 'type': 1},
    {'method': plot_levels, 'gas': 'HNO3', 'var': 'n', 'type': 1},
    {'method': plot_levels, 'gas': 'HNO3', 'var': 'Tatmxavk', 'type': 1},
    # Tatm
    {'method': plot_levels, 'gas': 'Tatm', 'var': 'avk', 'type': 1},
    {'method': plot_levels, 'gas': 'Tatm', 'var': 'n', 'type': 1},
]

for plot in plots:
    print('plotting', plot)
    df = import_data('data/motiv/error-estimation/METOPA_20160201001156_48180_20190323165817.csv',
                     gas=plot['gas'],
                     var=plot['var'])
    method = plot.pop('method')
    method(df, **plot)
    del df
