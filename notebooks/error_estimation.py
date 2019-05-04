# %% [markdown]
# # Error Estimation
# %%
import glob

import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluationErrorEstimation

thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1]


def altitude_by(level_of_interest: int):
    # TODO verify altitudes by mattias
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


def import_data(path_pattern: str, gas: str = None, var: str = None) -> pd.DataFrame:
    frames = []
    for file in glob.glob(path_pattern):
        frame = pd.read_csv(file, index_col=None, header=0)
        # filter to gas and variables
        if gas:
            frame = frame[frame['gas'] == gas]
        if var:
            frame = frame[frame['var'] == var]
        # set threshold == 0 to 1 for consistency
        frame['threshold'].replace(0, 1, inplace=True)
        if 'level_of_interest' in frame:
            # set altitudes
            frame['altitude'] = frame['level_of_interest'].apply(altitude_by)
        if 'size' in frame:
            frame['size'] = frame['size'] / 1000
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True)


# %%
err_winter = import_data(
    'data/motiv/error-estimation/METOPA_20160201001156_48180_20190323165817.csv')
# %%
# err_summer = import_data('data/scc/error-estimation/METOP*_20160801*.csv')
# %%
size_winter = import_data(
    'data/motiv/compression-summary/METOP*_20160201*.csv')
# %%
# size_summer = import_data('data/scc/compression-size/METOP*_20160801*.csv')

# %% [markdown]
# Season: Winter

# %%
err_season = err_winter
# %%
size_season = size_winter

# %%


def filter_by(df: pd.DataFrame, gas: str, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['gas'] == gas) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


def plot_error_estimation_for(df, gas: str, var: str, level_of_interest: int):
    filtered_events = filter_by(df, gas, var, level_of_interest)
    ax = filtered_events.groupby('threshold')[
        'err'].mean().plot.bar(logy=True, rot=0)
    mean_error = filter_by(df, gas, var, level_of_interest, False)[
        'err'].mean()
    # set ylim to make line for cov error visible
    ax.set_ylim(top=mean_error * 5)
    ax.axhline(mean_error, color='red', label='Error')
    ax.invert_xaxis()
    plt.xticks(np.arange(5), map(lambda t: f'{t:.0e}', thresholds))
    ax.set_ylabel('Reconstruction error')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.title(f'Error estimation for {gas} {var} at {level_of_interest}')
    plt.show()


def plot_levels(df: pd.DataFrame, gas: str, var: str, type):
    ax = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['type'] == type)
    ].groupby(['threshold', 'altitude']).mean()['err'].unstack().plot.bar(logy=True, rot=0)
    ax.invert_xaxis()
    plt.xticks(np.arange(5), map(lambda t: f'{t:.0e}', thresholds))
    plt.title(f'Error estimation for {gas} {var} type {type}')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.show()


def plot_types(df: pd.DataFrame, gas: str, var: str, level_of_interest: int):
    ax = df[
        (df['gas'] == gas) &
        (df['var'] == var) &
        (df['level_of_interest'] == level_of_interest)
    ].groupby(['threshold', 'type']).mean()['err'].unstack().plot.bar(logy=True, rot=0, legend='Type')
    plt.title(
        f'Error estimation {gas} {var} with level of interest {level_of_interest}')
    ax.invert_xaxis()
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.xticks(np.arange(5), map(lambda t: f'{t:.0e}', thresholds))
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
        'size'].plot.bar(rot=0, legend=False)
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


# %%[markdown]
# # Water Vapour
#
# ### Averaging Kernel
plot_size_for(size_season, 'WV', 'avk')
# %%
plot_levels(err_season, 'WV', 'avk', 1)
# %%
plot_levels(err_season, 'WV', 'avk', 2)
# %%
plot_types(err_season, 'WV', 'avk', -19)
# %%
plot_types(err_season, 'WV', 'avk', -16)

# %%[markdown]
# ## Noise Matrix
plot_size_for(size_season, 'WV', 'n')
# %%
plot_levels(err_season, 'WV', 'n', 1)
# %%
plot_levels(err_season, 'WV', 'n', 2)
# %%
plot_types(err_season, 'WV', 'n', -19)
# %%
plot_types(err_season, 'WV', 'n', -16)

# %%[markdown]
# ## Cross averaging kernels with atmospheric temperature
plot_size_for(size_season, 'WV', 'Tatmxavk')
# %% [markdown]
plot_levels(err_season, 'WV', 'Tatmxavk', 1)
# %% [markdown]
plot_levels(err_season, 'WV', 'Tatmxavk', 2)
# %%
plot_types(err_season, 'WV', 'Tatmxavk', -19)
# %%
plot_types(err_season, 'WV', 'Tatmxavk', -16)

# %% [markdown]
# # Greenhouse gases
# ## Averaging kernel
plot_size_for(size_season, 'GHG', 'avk')

# %%
plot_levels(err_season, 'GHG', 'avk', 1)

# %% [markdown]
# ## Noise matrix
plot_size_for(size_season, 'GHG', 'n')

# %%
plot_levels(err_season, 'GHG', 'n', 1)
# %% [markdown]
# ## Cross averaging kernel
plot_size_for(size_season, 'GHG', 'Tatmxavk')
# %%
plot_levels(err_season, 'GHG', 'Tatmxavk', 1)

# %% [markdown]
# # Nitrid Acid

plot_size_for(size_season, 'HNO3', 'avk')
# %%
plot_levels(err_season, 'HNO3', 'avk', type=1)
# %%
plot_size_for(size_season, 'HNO3', 'Tatmxavk')
# %%
plot_levels(err_season, 'HNO3', 'Tatmxavk', type=1)
# %%
plot_size_for(size_season, 'HNO3', 'n')
# %%
plot_levels(err_season, 'HNO3', 'n', type=1)

# %% [markdown]
# # Atmospheric Temperature
# %%
plot_size_for(size_season, 'Tatm', 'avk')
# %%
plot_levels(err_season, 'Tatm', 'avk', type=1)
# %%
plot_size_for(size_season, 'Tatm', 'n')
# %%
plot_levels(err_season, 'Tatm', 'n',  type=1)
