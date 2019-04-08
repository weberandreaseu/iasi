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
    gases=['WV', 'GHG', 'HNO3'],
    variables=['avk', 'n', 'Tatmxavk']
    # variables=['avk']
)

assert luigi.build([task], local_scheduler=True)

types = {'event': np.int, 'level_of_interest': np.int, 'err': np.float,
         type: np.int, 'rc_error': bool, 'threshold': np.float, 'var': str}
wv = pd.read_csv(task.output()['WV'].path, dtype=types)
ghg = pd.read_csv(task.output()['GHG'].path, dtype=types)
nho3 = pd.read_csv(task.output()['HNO3'].path, dtype=types)


def filter_by(df: pd.DataFrame, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


def plot_error_estimation_for(df, gas: str, var: str, level_of_interest: int):
    filtered_events = filter_by(df, var, level_of_interest)
    ax = filtered_events.groupby('threshold')[
        'err'].mean().plot.bar(logy=True, rot=0)
    mean_error = filter_by(df, var, level_of_interest, False)['err'].mean()
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

# %% [markdown]
# # Greenhouse gases

plot_error_estimation_for(ghg, 'GHG', 'avk', -10)
# %%
plot_error_estimation_for(ghg, 'GHG', 'Tatmxavk', -10)
# %%
plot_error_estimation_for(ghg, 'GHG', 'n', -10)

# %% [markdown]
# # Nitrid Acid

plot_error_estimation_for(nho3, 'NHO3', 'avk', -6)
# %%
plot_error_estimation_for(nho3, 'NHO3', 'Tatmxavk', -6)
# %%
plot_error_estimation_for(nho3, 'NHO3', 'n', -6)
