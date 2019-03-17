# %%
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iasi.evaluation import EvaluateCompression

task = EvaluateCompression(
    # force=True,
    dst='/tmp/iasi',
    file='data/input/MOTIV-slice-1000.nc',
    # file='test/resources/MOTIV-single-event.nc',
    variable='state/WV/atm_avk'
)

assert luigi.build([task], local_scheduler=True)

df = pd.read_csv(task.output().open('r'), dtype={'compressed': bool})
df.head()

# %%
df.plot.bar(x='threshold', y='size', rot=0)
plt.title('Data size reduction')
plt.show()

# %%
df.plot.bar(x='threshold', y=['diff_min', 'diff_mean', 'diff_max',
                              'diff_apost_min', 'diff_apost_mean', 'diff_apost_max'], logy=True, rot=0)
plt.show()
