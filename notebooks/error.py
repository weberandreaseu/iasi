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
    gases=['WV'],
    variables=['atm_avk', 'atm_n']
)

assert luigi.build([task], local_scheduler=True)

types = {'event': np.int, 'level_of_interest': np.int, 'err': np.float,
         type: np.int, 'rc_error': bool, 'threshold': np.float, 'var': str}
df = pd.read_csv(task.output()['WV'].path, dtype=types)


def filter_by(df: pd.DataFrame, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


def plot_error_estimation_for(gas: str, var: str, level_of_interest: int):
    filtered_events = filter_by(df, var, level_of_interest)
    ax = filtered_events.groupby('threshold')[
        'err'].mean().plot.bar(logy=True, rot=0)
    mean_error = filter_by(df, var, level_of_interest, False)['err'].mean()
    # set ylim to make line for cov error visible
    ax.set_ylim(top=1)
    ax.axhline(mean_error, color='red', label='Error')
    ax.set_ylabel('Reconstruction error')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.title(f'Error estimation for {gas} {var} at {level_of_interest}')
    plt.show()


df.head()

# %%[markdown]
# ## Water Vapour
#
# ### Averaging Kernel
# - Level of interest: -16
plot_error_estimation_for('WV', 'atm_avk', -16)

# %%[markdown]
# ### Noise Matrix
# - Level of interest: -16
plot_error_estimation_for('WV', 'atm_n', -16)
