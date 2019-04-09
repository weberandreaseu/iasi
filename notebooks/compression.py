# %%
import matplotlib
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluationCompressionSize

task = EvaluationCompressionSize(
    # force_upstream=True,
    dst='data',
    file='data/input/METOPA_20160625001453_50240_20190209151722.nc',
    # file='test/resources/MOTIV-single-event.nc',
    # file='data/input/MOTIV-slice-1000.nc',
    gases=['WV', 'GHG', 'HNO3', 'Tatm'],
    variables=['avk', 'n', 'Tatmxavk']
)

assert luigi.build([task], local_scheduler=True)

df = pd.read_csv(task.output().open('r'))
original = df[df['ancestor'] == 'MoveVariables']
compressed = df[df['ancestor'] == 'CompressDataset']


def filter_by(df, gas: str, variable: str, threshold: float = None):
    return df[(df['gas'] == gas) & (df['variable'] == variable)]


def plot_size_for(gas: str, variable: str):
    gas_compressed = filter_by(compressed, gas, variable)
    ax = gas_compressed.plot.bar(x='threshold', y='size', rot=0, legend=False)
    gas_original = filter_by(original, gas, variable)
    assert len(gas_original) == 1
    original_size = gas_original['size'].values[0]
    ax.axhline(original_size, color='red')
    ax.text(0, original_size * 0.94, 'Original size',
            horizontalalignment='center')
    # ax.legend(loc='upper left')
    ax.set_ylabel('File size in kB')
    ax.set_xlabel('Threshold for eigenvalue selection')
    plt.title(f'Data size reduction for {gas} {variable}')
    plt.show()


def plot_total_size_for(gas: str):
    gas_compressed = compressed[compressed['gas'] == gas]
    ax = gas_compressed.groupby('threshold').sum()['size'].plot.bar(rot=0)
    ax.set_ylabel('Size in kB')
    ax.set_xlabel('Threshold for eigenvalue selection')
    gas_original_sum = original[original['gas'] == gas]['size'].sum()
    ax.axhline(gas_original_sum, color='red')
    ax.text(0, gas_original_sum * 0.94, 'Original size',
            horizontalalignment='center')
    plt.title(f'Total size for {gas}')
    plt.show()


# %%
plot_size_for('WV', 'avk')

# %%
plot_size_for('WV', 'n')

# %%
plot_size_for('WV', 'Tatmxavk')
# %%
plot_size_for('GHG', 'avk')
# %%
plot_size_for('GHG', 'n')
# %%
plot_size_for('GHG', 'Tatmxavk')
# %%
plot_size_for('HNO3', 'avk')
# %%
plot_size_for('HNO3', 'n')
# %%
plot_size_for('HNO3', 'Tatmxavk')
# %%
plot_size_for('Tatm', 'avk')
# %%
plot_size_for('Tatm', 'n')
# %%
plot_total_size_for('WV')
# %%
plot_total_size_for('GHG')
# %%
plot_total_size_for('HNO3')
# %%
plot_total_size_for('Tatm')
