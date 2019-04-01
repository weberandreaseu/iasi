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
    variables=['atm_avk']
)

assert luigi.build([task], local_scheduler=True)

types = {'event': np.int, 'level_of_interest': np.int, 'err': np.float,
         type: np.int, 'rc_error': bool, 'threshold': np.float, 'var': str}
df = pd.read_csv(task.output()['WV'].path, dtype=types)
df.head()


def filter_by(df: pd.DataFrame, var: str, level_of_interest: int, rc_error=True):
    return df[(df['rc_error'] == rc_error) & (df['var'] == var) & (df['level_of_interest'] == level_of_interest)]


# %%
wv_avk_10 = filter_by(df, 'atm_avk', -10)
ax = wv_avk_10.groupby('threshold')['err'].mean().plot.bar(logy=True)
wv_avk_10_mean_error = filter_by(df, 'atm_avk', -10, False)['err'].mean()
ax.set_ylim(top=wv_avk_10_mean_error + 0.5)
ax.axhline(wv_avk_10_mean_error, color='red')
ax.set_ylabel('File size in kB')
ax.set_xlabel('Threshold for eigenvalue selection')
plt.title(f'Data size reduction for {task.variables[0]}')
plt.show()

# %%
ax = df.plot.bar(x='threshold', y='size', rot=0, legend=False)
# ax.legend(loc='upper left')
ax.set_ylabel('File size in kB')
ax.set_xlabel('Threshold for eigenvalue selection')
plt.title(f'Data size reduction for {task.variables[0]}')
plt.show()
