# %%
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

nc = Dataset('test/resources/MOTIV-single-event.nc')
nol = 28

alt = nc['atm_altitude'][0, :nol].data
# altitude tropopause
alt_trop = 9486.431
# altitude stratosphere TODO end of stratosphere?
alt_strat = 25000

sig = np.ndarray(nol)

for i in range(nol):
    # below tropopause
    if alt[i] < alt_trop:
        sig[i] = 2500 + (alt[i] - alt[0]) * ((5000-2500)/(alt_trop-alt[0]))
    # inside statrophere
    if alt[i] >= alt_trop and alt[i] < alt_strat:
        sig[i] = 5000+(alt[i]-alt_trop)*((10000-5000)/(alt_strat-alt_trop))
    # above stratosphere
    if alt[i] > alt_strat:
        sig[i] = 10000


def assumed_covariance(a: np.ndarray, sig: np.ndarray) -> np.ndarray:
    sa = np.ndarray((nol, nol))
    for i in range(nol):
        for j in range(nol):
            sa[i, j] = a[i] * a[j] * \
                np.exp(-((alt[i] - alt[j])*(alt[i] - alt[j])) /
                       (2 * sig[i] * sig[j]))
    return sa


# CH4
a_CH4 = np.ndarray((nol))
for i in range(nol):
    if alt[i] < alt_trop:
        a_CH4[i] = 100
    if alt[i] >= alt_trop:
        a_CH4[i] = 100 + (alt[i] - alt_trop) * \
            ((250 - 100)/(alt_strat - alt_trop))
    if alt[i] >= alt_strat:
        a_CH4[i] = 250

# HNO3
a_HNO3 = np.ndarray((nol, nol))

# sa_HNO3 = assumed_covariance(None, sig * 1.2)
sa_CH4 = assumed_covariance(a_CH4, sig * 0.6)


plt.imshow(sa_CH4)
plt.show()


# %%
