# %% [markdown]
# # Error Estimation
# %%
import matplotlib
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluationErrorEstimation

task = EvaluationErrorEstimation(
    force_upstream=True,
    dst='data',
    file='data/input/MOTIV-slice-100.nc',
    # file='test/resources/MOTIV-single-event.nc',
    gases=['WV', 'GHG', 'HNO3', 'Tatm'],
    variables=['avk', 'n', 'Tatmxavk'],
    thresholds=[1e-2, 1e-3, 1e-4, 1e-5]
)

assert luigi.build([task], local_scheduler=True)

types = {'event': np.int, 'level_of_interest': np.int, 'err': np.float,
         type: np.int, 'rc_error': bool, 'threshold': np.float, 'var': str, 'gas': str}

with task.output().open() as file:
    df = pd.read_csv(file, dtype=types)
    wv = df[df['gas'] == 'WV']
    ghg = df[df['gas'] == 'GHG']
    hno3 = df[df['gas'] == 'HNO3']
    tatm = df[df['gas'] == 'Tatm']


def filter_by(df: pd.DataFrame, gas: str, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['gas'] == gas) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


def plot_error_estimation_for(df, gas: str, var: str, level_of_interest: int):
    filtered_events = filter_by(df, gas, var, level_of_interest)
    ax = filtered_events.groupby('threshold')[
        'err'].mean().plot.bar(logy=True, rot=0)
    mean_error = filter_by(df, gas, var, level_of_interest, False)['err'].mean()
    # set ylim to make line for cov error visible
    ax.set_ylim(top=mean_error * 5)
    ax.axhline(mean_error, color='red', label='Error')
    ax.set_ylabel('Reconstruction error')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.title(f'Error estimation for {gas} {var} at {level_of_interest}')
    plt.show()


wv.head()

# %%[markdown]
# ## Water Vapour
#
# ### Averaging Kernel
# - Level of interest: -16
plot_error_estimation_for(wv, 'WV', 'avk', -16)

# %%[markdown]
# ### Noise Matrix
# - Level of interest: -16
plot_error_estimation_for(wv, 'WV', 'n', -16)
# %%[markdown]
# ### Crosse averaging temperature in atmosphere
# - Level of interest: -16
plot_error_estimation_for(wv, 'WV', 'Tatmxavk', -16)

# %%
wv[
    (wv['var'] == 'avk') &
    (wv['level_of_interest'] == -16)
].groupby(['threshold', 'type']).mean()['err'].unstack().plot.bar(logy=True, rot=0)
plt.show()

# %%
wv[
    (wv['var'] == 'avk') &
    (wv['type'] == 1)
].groupby(['threshold', 'level_of_interest']).mean()['err'].unstack().plot.bar(logy=True, rot=0)
plt.title('Error estimation for WV avk type 1')
plt.show()

# %% [markdown]
# # Greenhouse gases

plot_error_estimation_for(ghg, 'GHG', 'avk', -10)
# %%
plot_error_estimation_for(ghg, 'GHG', 'Tatmxavk', -10)
# %%
plot_error_estimation_for(ghg, 'GHG', 'n', -10)

# %% [markdown]
# # Nitrid Acid

plot_error_estimation_for(hno3, 'HNO3', 'avk', -6)
# %%
plot_error_estimation_for(hno3, 'HNO3', 'Tatmxavk', -6)
# %%
plot_error_estimation_for(hno3, 'HNO3', 'n', -6)

# %% [markdown]
# # Atmospheric Temperature
plot_error_estimation_for(tatm, 'Tatm', 'avk', -19)
# %%
plot_error_estimation_for(tatm, 'Tatm', 'n', -19)
