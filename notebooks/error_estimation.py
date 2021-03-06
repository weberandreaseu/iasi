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
                inlcude_coordinates=False,
                rank: str = None) -> pd.DataFrame:
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
            if rank:
                target_rank = nc[rank][...]
                coordinates = pd.DataFrame(
                    {'lat': lat, 'lon': lon, 'rank': target_rank})
            else:
                coordinates = pd.DataFrame(
                    {'lat': lat, 'lon': lon, 'rank': rank})
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
        # norm=colors.LogNorm(df.err.min(), df.err.max()),
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
    plt.savefig('avk_winter_summer.pdf', bbox_inches='tight')


def plot_size_season(df: pd.DataFrame, gas: str, var: str):
    df = df[
        (df['gas'] == gas) &
        (df['variable'] == var)
    ]
    groups = df.groupby(['threshold'])
    original_size = 0
    for i, (name, group) in enumerate(groups):
        if i == 0:
            original_size = group.mean()['size']
        current_size = group.mean()['size']
        print(f'{gas}, {var} Tresh: {name}, orginal: {original_size}, curr {current_size}, ratio {(current_size / original_size) * 100}')
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
    ax.set_xlabel('threshold')
    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    plt.savefig(f'size_{gas}_{var}.pdf', bbox_inches='tight')
    # plt.title(f'Data size reduction for {gas} {var}')
    plt.close()


def plot_levels(df: pd.DataFrame, gas: str, var: str, type, season=None):
    df = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['type'] == type)
    ]
    print(df.groupby(['threshold', 'altitude']).mean()['err'])

    # plot reconstruction error
    original = df[df['rc_error'] == False]
    compressed = df[df['rc_error'] == True]
    groups = compressed.groupby(['threshold', 'altitude']).mean()['err']
    ax = groups.unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    ax.invert_xaxis()

    # plot original error
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, err in enumerate(original.groupby(['altitude']).mean()['err']):
        ax.axhline(err, color=colors[index], linestyle='--')

    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    # plt.title(f'Error estimation for {gas} {var} type {type}')
    ax.set_xlabel('threshold')
    plt.ylabel('error variance [log(ppmv)^2]')
    plt.savefig(f'err_{gas}_{var}_type{type}.pdf', bbox_inches='tight')
    plt.close()


def plot_types(df: pd.DataFrame, gas: str, var: str, level_of_interest: int):
    df = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['level_of_interest'] == level_of_interest)
    ]
    original = df[df['rc_error'] == False]
    compressed = df[df['rc_error'] == True]
    ax = compressed.groupby(['threshold', 'type']).mean()[
        'err'].unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    # plt.title(
    #     f'Error estimation {gas} {var} with level of interest {level_of_interest}')
    ax.invert_xaxis()
    ax.set_xlabel('threshold')
    ax.set_ylabel('error variance [log(ppmv)^2]')

    # plot original error
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, mean in enumerate(original.groupby(['type']).mean()['err']):
        ax.axhline(mean, color=colors[index], linestyle='--')

    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    plt.savefig(f'err_{gas}_{var}_loi{level_of_interest}.pdf',
                bbox_inches='tight')
    plt.close()


# %% [markdown]

# ## Example: Water Vapour Averaging Kernel
# plot size
# size = import_size('data/final/size/METOP*.csv')
# plot_size_season(size, 'WV', 'avk')

# %%
# err = import_data('data/final/WV/avk/METOPB_20160801115357_20086_20190312071833.nc',
#                   inlcude_coordinates=True)
# plot_levels(err, 'WV', 'avk', type=1)

# %%
# plot_types(err, 'WV', 'avk', -19)


# %%close()
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
    if plot['gas'].startswith('GHG'):
        df = import_data(f'data/final/GHG/{plot["var"]}/*.nc',
                         gas=plot['gas'],
                         var=plot['var'])
    else:
        df = import_data(f'data/final/{plot["gas"]}/{plot["var"]}/*.nc',
                         gas=plot['gas'],
                         var=plot['var'])
    print('start plotting')
    method = plot.pop('method')
    method(df, **plot)
    print('finished plotting')
    del df


# %%

size = import_size('data/final/size/*.csv')
variables = {
    ('WV', 'avk'),
    ('WV', 'Tatmxavk'),
    ('WV', 'n'),
    ('GHG', 'avk'),
    ('GHG', 'Tatmxavk'),
    ('GHG', 'n'),
    ('HNO3', 'avk'),
    ('HNO3', 'Tatmxavk'),
    ('HNO3', 'n'),
    ('Tatm', 'avk'),
    ('Tatm', 'n')
}

for gas, var in variables:
    plot_size_season(size, gas, var)


# %%
def import_eigenvalues(pattern, var) -> pd.DataFrame:
    frames = []
    for file in glob.glob(pattern):
        nc = Dataset(file)
        k = nc['/state/WV/avk/k'][...]
        lat = nc['lat'][...]
        lon = nc['lon'][...]
        nc.close()
        frame = pd.DataFrame({'lat': lat, 'lon': lon, 'k': k})
        frame = frame[frame['k'].between(0, 58)]
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True)


# %%
def plot_rank_map(summer: pd.DataFrame, winter: pd.DataFrame):
    min_err = min(summer['k'].min(), winter['k'].min())
    max_err = max(summer['k'].max(), winter['k'].max())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    sc1 = ax1.scatter(
        summer.lon,
        summer.lat,
        c=summer['k'],
        marker='.',
        s=1,
        # log scale for error
        # norm=colors.LogNorm(min_err, max_err),
        vmin=min_err,
        vmax=max_err,
        transform=ccrs.PlateCarree())
    # plt.colorbar(sc1)
    ax1.coastlines()
    ax1.set_title('2016-08-01')
    sc2 = ax2.scatter(
        winter.lon,
        winter.lat,
        c=winter['k'],
        marker='.',
        s=1,
        # log scale for error
        # norm=colors.LogNorm(min_err, max_err),
        vmin=min_err,
        vmax=max_err,
        transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_title('2016-02-01')
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(sc2, cax=cbar_ax)
    plt.savefig('rank_winter_summer.pdf', bbox_inches='tight')


# %%

rank_summer = import_eigenvalues(
    'data/eigenvalues/METOP?_201608*.nc', 'state/WV/avk/k')
rank_winter = import_eigenvalues(
    'data/eigenvalues/METOP?_201602*.nc', 'state/WV/avk/k')


# %%
plot_rank_map(rank_summer, rank_winter)

# %%
err = import_data('data/final/WV/avk/*.nc',
                  gas='WV',
                  var='avk',
                  inlcude_coordinates=True,
                  loi=-19,
                  type=1)

err_rc = err[err['threshold'] == 0.001]
err_org = err[err['rc_error'] == False]
assert err_org.shape == err_rc.shape

# %%
plt.scatter(err_rc.err, err_org.err)
plt.ylim(err_org.err.min(), err_org.err.max())
plt.xlim(err_rc.err.min(), err_rc.err.max())
# %%


def get_outlier(df):
    return df[(df.err-df.err.mean()).abs() > 3*df.err.std()]


outlier_rc = get_outlier(err_rc)
plot_error_map(outlier_rc)

outlier_org = get_outlier(err_org)
plot_error_map(outlier_org)

# %%
min, max = err_rc.err.min(),  err_rc.err.max()
fig = plt.figure(figsize=(4, 4))
plt.hist(err_rc.err, bins=np.logspace(np.log10(min), np.log10(max), num=60), orientation='horizontal')
plt.gca().set_yscale("log")
plt.savefig('wv_avk_rc_err_dist.pdf', bbox_inches='tight')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('rc error variance [log(ppmv)^2]')
plt.xlabel('# measurements')
plt.savefig('wv_avk_rc_err_dist.pdf', bbox_inches='tight')
plt.show()


# %%
err = import_data('data/final/WV/avk/*.nc',
                  rank='state/WV/avk/k',
                  gas='WV',
                  var='avk',
                  inlcude_coordinates=True,
                  loi=-19,
                  type=1)
err_rc = err[err['threshold'] == 0.001]

# %%
groups = err_rc.groupby(['rank'])['err']
means = groups.mean()
errors = groups.std()

fig, ax = plt.subplots()
means.plot.bar(yerr=([0] * 15, errors.tolist()), ax=ax, capsize=4)

# %%
# err_rc = err[err['threshold'] == 0.001]
ax = err_rc.boxplot(column=['err'], by='rank', figsize=(6, 4))
fig = ax.get_figure()
fig.suptitle(None)
plt.title(None)
plt.ylabel('rc error variance [log(ppmv)^2]')
plt.xlabel('target rank')
plt.yscale('log')
plt.savefig('err_box_by_rank_wv_avk.pdf', bbox_inches='tight')