# %% [markdown]
# # Evaluation of Singular Value Decomposition

# Singular Value Decomposition (SVD) uses `U, s, Vh`.


# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import luigi
from iasi import DeltaDRetrieval
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# disable logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %%
file = 'data/input/IASI-A_20160627_50269_v2018_fast_part0_20180413215350.nc'
direct = DeltaDRetrieval(
    dst='./data',
    file=file,
    dim=28
)

svd = [DeltaDRetrieval(
    dst='./data',
    file=file,
    svd=True,
    dim=d
) for d in range(2, 29)]

luigi.build([direct] + svd, workers=2)

# %%


def read_task_output(task) -> pd.DataFrame:
    with task.output().open('r') as file:
        df = pd.read_csv(file, usecols=['H2O', 'delD'])
        df['dim'] = task.dim if task.dim else 28
        size = len(df.index)
        df = df.dropna()
        dropped = size - len(df.index)
        print(f'Dropped {dropped} rows from  dataframe with dim {task.dim}')
        return df.reset_index().set_index(['dim', 'index'])


# %%
original = read_task_output(direct)
df = pd.DataFrame()
for task in svd:
    new_df = read_task_output(task)
    # output of each task should be the same
    assert new_df.shape == original.shape
    df = df.append(new_df)

# %%
metrics = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score
}
H2O_error = pd.DataFrame(columns=metrics)
delD_error = pd.DataFrame(columns=metrics)
# iterate over dimensions
for dim, group in df.groupby(level=[0]):
    H2O_error.loc[dim] = [
        metric(group['H2O'], original['H2O'])
        for metric in metrics.values()
    ]
    delD_error.loc[dim] = [
        metric(group['delD'], original['delD'])
        for metric in metrics.values()
    ]

# %%
H2O_error


# %%
delD_error

# %%
delD_error.mae.plot()
plt.show()


H2O_error.mae.plot()
plt.show()
