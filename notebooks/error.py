# %%
import matplotlib
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluationErrorEstimation

task = EvaluationErrorEstimation(
    force=True,
    dst='data',
    file='data/input/MOTIV-slice-100.nc',
    # file='test/resources/MOTIV-single-event.nc',
    gases=['WV'],
    variables=['atm_avk']
)

assert luigi.build([task], local_scheduler=True)

df = pd.read_csv(task.output()['WV'].path)
df.head()

#%%
df.groupby('threshold')['err'].mean().plot.bar()

# %%
ax = df.plot.bar(x='threshold', y='size', rot=0, legend=False)
# ax.legend(loc='upper left')
ax.set_ylabel('File size in kB')
ax.set_xlabel('Threshold for eigenvalue selection')
plt.title(f'Data size reduction for {task.variables[0]}')
plt.show()
