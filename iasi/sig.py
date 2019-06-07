# %%
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

nc = Dataset('test/resources/MOTIV-single-event.nc')
nol = 28

alt = nc['atm_altitude'][0, :nol].data
# altitude tropopause
alt_trop = 9486.431
# alt_trop = 3800
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
a_HNO3 = np.ndarray((nol))
for i in range(nol):

    # surface is more than 4km below tropopause
    if alt[0] < alt_trop - 4000:
        # TODO: template is strage here. ask for feedback
        if alt[i] < alt_trop - 4000:
            a_HNO3[i] = 2400 + (alt[i] - alt[0]) * \
                ((1200 - 2400)/(alt_trop - 4000 - alt[0]))
        if alt_trop - 4000 <= alt[i] < alt_trop + 8000:
            a_HNO3[i] = 1200
        if alt_trop + 8000 <= alt[i] < 50000:
            a_HNO3[i] = 1200 + (alt[i] - (alt_trop + 8000)) * \
                ((300-1200) / (50000 - (alt_trop + 8000)))
        if alt[i] >= 50000:
            a_HNO3[i] = 300
    else:
        if alt_trop - 4000 < alt[i] < alt_trop + 8000:
            a_HNO3[i] = 1200
        if alt_trop + 8000 < alt[i] < 50000:
            a_HNO3[i] = 1200 + (alt[i] - (alt_trop + 8000)) * \
                ((300 - 1200)/(50000 - (alt_trop + 8000)))
        if alt[i] >= 50000:
            a_HNO3[i] = 300

sa_HNO3 = assumed_covariance(a_HNO3, sig * 1.2)
sa_CH4 = assumed_covariance(a_CH4, sig * 0.6)


plt.imshow(sa_HNO3)
plt.show()
plt.plot(a_HNO3)
plt.show()


# %%
# plt.plot(a_HNO3)


# %%
