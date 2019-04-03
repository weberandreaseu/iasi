# %%
import matplotlib
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluationCompressionSize

task = EvaluationCompressionSize(
    force=True,
    dst='data',
    file='data/input/MOTIV-slice-1000.nc',
    # file='test/resources/MOTIV-single-event.nc',
    gases=['WV'],
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


# %%
plot_size_for('WV', 'avk')

# %%
plot_size_for('WV', 'n')

# %%
plot_size_for('WV', 'Tatmxavk')

# %%
# Total size
wv = compressed[compressed['gas'] == 'WV']
ax = wv.groupby('threshold').sum()['size'].plot.bar(rot=0)
ax.set_ylabel('Size in kB')
ax.set_xlabel('Threshold for eigenvalue selection')
gas_original_sum = original[original['gas'] == 'WV']['size'].sum()
ax.axhline(gas_original_sum, color='red')
ax.text(0, gas_original_sum * 0.94, 'Original size',
        horizontalalignment='center')
plt.title('Total size for WV')
plt.show()
