# %% [markdown]
# # Error Estimation
# %%
import glob
import logging
import os

import cartopy.crs as ccrs
import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 0]


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
                inlcude_coordinates=True) -> pd.DataFrame:
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
        if threshold:
            frame = frame[frame['threshold'] == threshold]
        # set threshold == 0 to 1 for consistency
        frame['threshold'].replace(0, 1, inplace=True)
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


# %%
# def aggregate_error(path_pattern: str):
#     for file in glob.glob(path_pattern):
#         df = import_data(file)
#         aggregated = df.groupby(
#             ['gas', 'var', 'threshold', 'level_of_interest', 'type'])['err'].mean()
#         path, filename = os.path.split(file)
#         target = os.path.join(path, 'aggregated', filename)
#         aggregated.to_csv(
#             target, header=['err'], index_label=['gas', 'var', 'threshold', 'level_of_interest', 'type'])


# aggregate_error('data/motiv/error-estimation/*.csv')

# # %%


# def load_aggregated(path_pattern: str) -> pd.DataFrame:
#     frames = []
#     for file in glob.glob(path_pattern):
#         frame = pd.read_csv(file, index_col=None, header=0)
#         frames.append(frame)
#     return pd.concat(frames, axis=0, ignore_index=True)


# %%

file = 'data/motiv/error-estimation/METOPA_20160201001156_48180_20190323165817.csv'
df = import_data(file, gas='WV', var='avk',
                 inlcude_coordinates=False, loi=-6, threshold=1e-3, type=1)


# %%
err_winter = import_data(
    'data/motiv/error-estimation/METOP*_20160201*.csv', 'Tatm')
# %%
# err_summer = import_data(
#     'data/motiv/error-estimation/METOP*_20160801*.csv', 'Tatm')
# %%
# size_winter = import_data(
#     'data/motiv/compression-summary/METOP*_20160201*.csv')
# %%
size = import_data(
    'data/motiv/compression-summary/METOP*_2016*.csv')

# %% [markdown]
# Season: Winter

# %%


def filter_by(df: pd.DataFrame, gas: str, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['gas'] == gas) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


def plot_levels(df: pd.DataFrame, gas: str, var: str, type, season=None):
    ax = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['type'] == type)
    ].groupby(['threshold', 'altitude']).mean()['err'].unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    ax.invert_xaxis()
    plt.xticks(np.arange(5), map(lambda t: f'{t:.0e}', thresholds))
    # plt.title(f'Error estimation for {gas} {var} type {type}')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.savefig(f'err_{gas}_{var}_type{type}.pdf')
    plt.show()


def plot_types(df: pd.DataFrame, gas: str, var: str, level_of_interest: int):
    ax = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['level_of_interest'] == level_of_interest)
    ].groupby(['threshold', 'type']).mean()['err'].unstack().plot.bar(logy=True, rot=0, figsize=(4, 4))
    # plt.title(
    #     f'Error estimation {gas} {var} with level of interest {level_of_interest}')
    ax.invert_xaxis()
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.xticks(np.arange(5), map(lambda t: f'{t:.0e}', thresholds))
    plt.savefig(f'err_{gas}_{var}_loi{level_of_interest}.pdf')
    plt.show()


def plot_size_for(df: pd.DataFrame, gas: str, var: str):
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
    ax = compressed.groupby(['threshold']).mean()[
        'size'].plot.bar(rot=0, legend=False, figsize=(4, 4))
    original_mean = original['size'].mean()
    ax.axhline(original_mean, color='red')
    ax.invert_xaxis()
    ax.text(2.8, original_mean * 0.92, 'Mean original size',
            horizontalalignment='center')
    ax.set_ylabel('Mean size in MB')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.xticks(np.arange(4), map(lambda t: f'{t:.0e}', thresholds))
    plt.title(f'Data size reduction for {gas} {var}')
    plt.show()


# %%
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
    df = import_data('data/motiv/error-estimation/*.csv',
                     gas=plot['gas'],
                     var=plot['var'])
    method = plot.pop('method')
    method(df, **plot)
    del df


# %%[markdown]
# # Water Vapour
#
# ### Averaging Kernel
plot_size_for(size, 'WV', 'avk')
# %%
plot_levels(err_winter, 'WV', 'avk', 1)
# %%
plot_levels(err_summer, 'WV', 'avk', 2)
# %%
plot_types(err_winter, 'WV', 'avk', -19)
# %%
plot_types(err_summer, 'WV', 'avk', -16)

# %%[markdown]
# ## Noise Matrix
plot_size_for(size, 'WV', 'n')
# %%
plot_levels(err_winter, 'WV', 'n', 1)
# %%
plot_levels(err_summer, 'WV', 'n', 2)
# %%
plot_types(err_winter, 'WV', 'n', -19)
# %%
plot_types(err_summer, 'WV', 'n', -16)

# %%[markdown]
# ## Cross averaging kernels with atmospheric temperature
plot_size_for(size, 'WV', 'Tatmxavk')
# %% [markdown]
plot_levels(err_summer, 'WV', 'Tatmxavk', 1)
# %% [markdown]
plot_levels(err_summer, 'WV', 'Tatmxavk', 2)
# %%
plot_types(err_winter, 'WV', 'Tatmxavk', -19)
# %%
plot_types(err_summer, 'WV', 'Tatmxavk', -16)

# %% [markdown]
# # Greenhouse gases
# ## Averaging kernel
plot_size_for(size, 'GHG', 'avk')

# %%
plot_levels(err_winter, 'GHG_CH4', 'avk', 1)

# %% [markdown]
# ## Noise matrix
plot_size_for(size, 'GHG', 'n')

# %%
plot_levels(err_summer, 'GHG_CH4', 'n', 1, season='summer')
# %% [markdown]
# ## Cross averaging kernel
plot_size_for(size, 'GHG', 'Tatmxavk')
# %%
plot_levels(err_summer, 'GHG_CH4', 'Tatmxavk', 1, season='summer')

# %% [markdown]
# # Nitrid Acid

plot_size_for(size, 'HNO3', 'avk')
# %%
plot_levels(err_summer, 'HNO3', 'avk', type=1, season='summer')
# %%
plot_size_for(size, 'HNO3', 'Tatmxavk')
# %%
plot_levels(err_summer, 'HNO3', 'Tatmxavk', type=1, season='summer')
# %%
plot_size_for(size, 'HNO3', 'n')
# %%
plot_levels(err_summer, 'HNO3', 'n', type=1, season='summer')

# %% [markdown]
# # Atmospheric Temperature
# %%
plot_size_for(size, 'Tatm', 'avk')
# %%
plot_levels(err_summer, 'Tatm', 'avk', type=1, season='summer')
# %%
plot_size_for(size, 'Tatm', 'n')
# %%
plot_levels(err_summer, 'Tatm', 'n',  type=1, season='summer')
