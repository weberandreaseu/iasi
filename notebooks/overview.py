# %%[markdown]
# # Overview

# Some notes about output files of proffi_processing.

from os.path import abspath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy import stats
from sklearn.metrics import mean_absolute_error

# %%
nc = Dataset(
    'data/input/IASI-A_20160627_50269_v2018_fast_part0_20180413215350.nc')

# %%[markdown]
# ## Variables
#
# There are multiple compounds measured in atmosphere:
# - GHG: Profiles of green house gas species `{ln[N2O],ln[CH4]}` (Nitrous Oxide, Methane)
# - WV: Profiles of atmospheric water vapour species `{ln[H2O],ln[HDO]}` (Water, Hydrodeoxygenation)
# - HNO3: Profiles of `{HNO3}` (Nitrid Acid)
#
#
# In addition to profile, each compound has:
# - an a priori profile `p`
# - an averaging kernel matrix `avk`
# - a noise matrix `n`
#
#
# Furthermore, there are cross averaging kernels between:
# - Natural logarithm of temperature `{ln[T]}` respectively to `{GHG, WV, HNO3}`
#


class Compound:
    vars = [
        'state_{}atm',
        'state_{}atm_a',
        'state_{}atm_avk',
        'state_{}atm_n'
    ]

    def __init__(self, name: str = None, cross_kernels=[]):
        self.name = name
        if not name:
            self.vars = cross_kernels
        else:
            self.vars = [var.format(name) for var in self.vars]

    def minmax(self, var):
        data = var[:]
        data = data[~np.isnan(data)]
        return np.nanmin(data), np.nanmax(data)

    def describe(self, nc):
        row_format = "{:23}{:23}{:23.18f}{:23.18f}{:8}"
        for var in self.vars:
            min, max = self.minmax(nc[var])
            # last two dimensions are number of grid levels (28)? -> square!
            square = nc[var].dimensions[-2:] == (
                'atmospheric_grid_levels', 'atmospheric_grid_levels'
            )
            print(row_format.format(
                var,
                str(nc[var].shape[1:]),
                min,
                max,
                square
            ))


compounds = map(Compound, ['GHG', 'HNO3', 'WV', 'T'])
print(('{:23}'*4 + '{:8}').format('Variable', 'Shape', 'Min', 'Max', 'Square'))
for compound in compounds:
    compound.describe(nc)

cross_kernels = Compound(cross_kernels=[
    'state_Tatm2GHGatm_xavk',
    'state_Tatm2HNO3atm_xavk',
    'state_Tatm2WVatm_xavk'
])
cross_kernels.describe(nc)

# %%[markdown]
# ## Compression
#
# Our target is to reduce storage size. Because the measurements are already noisy/approximated,
# it is not absolutely necessary to keep all decimal places.
# Instead, we hope to get identical results with good approximations.
#
# The multi dimensional arrays containing most of values.
# Accordingly they have the highest potential for storage reduction.
# Arrays with higher dimension than two can be transformed to a matrix.
# E.g. `state_WVatm_n` with shape of $(2, 2, 28, 28)$ transforms to a $56 \times 56$ matrix.
# With that, we can apply matrix compression techniques to our data.
# Here we consinder only matrices with minimum size of $28 \times 28$ (marked as square in table above)
#
#
#
# ### Singular Value Decomposition
# - $m \times n$ matrices
#
#
# ### Eigen Decomposition
# - only square matrices

# %% [markdown]
# ## Floating point precision error of random variable
#
# Generate an uniform distributed array with values in different interval.


# %%
def diff_of_cast(a: np.ndarray):
    b = np.float32(a)
    return a - b


intervals, size = 5, 10000
a = np.ndarray((size, intervals))
label = []
for i in range(intervals):
    low = i**2
    high = (i+1)**2
    description = '[{}, {})'.format(low, high)
    values = np.random.uniform(low=low, high=high, size=size)
    diff = diff_of_cast(values)
    # print(description, stats.describe(diff))
    a[:, i] = diff
    label.append(description)
plt.boxplot(a)
plt.xticks(range(1, intervals + 1), label)
plt.show()


# %% [markdown]
# The results show, that floating point error grow with the interval.


# %%[markdown]

# ### Fast arithmetics
#
# TODO: analyse performance of numpy with 32 vs 64 bit values (AVX extensions?)
#
# TODO: analyse floating point error in arithmetic with single precision
#
# 1. For fast arithmetics, blas/lpack should be available. See in numpy `sysinfo`
# ```
# import numpy.distutils.system_info as sysinfomin
# sysinfo.show_all()
# ```
