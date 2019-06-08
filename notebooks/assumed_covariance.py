# %%
from netCDF4 import Dataset
import iasi.evaluation as eval
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset('test/resources/MOTIV-single-event.nc')
alt = nc['atm_altitude'][...]
nol = nc['atm_nol'][...]
alt_trop = nc['tropopause_altitude'][...]

wv = eval.WaterVapour('WV', nol, alt, None, alt_trop=alt_trop)
ghg = eval.GreenhouseGas('GHG', nol, alt, alt_trop=alt_trop)
hno3 = eval.NitridAcid('HNO3', nol, alt, alt_trop=alt_trop)

nol = nol.data[0]
alt = alt.data[0, :nol]
alt_trop = alt_trop.data[0]

def default_color(i: int):
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][i]

# %% [markdown]

# Water Vapour
cov = wv.assumed_covariance(0)
amp_H2O, amp_delD, sigma = wv.assumed_covariance(0, return_tuple=True)

plt.imshow(cov)
plt.colorbar()
plt.show()


fig, ax1 = plt.subplots()

# stratosphere
ax1.axhline(25, color='gray', linestyle='dotted')
ax1.text(1, 26, 'stratosphere',  color='gray')
# tropopause
ax1.axhline(alt_trop / 1000, color='gray', linestyle='dotted')
ax1.text(1, (alt_trop / 1000) + 1, 'tropopause', color='gray')
# 5km
ax1.axhline(5, color='gray', linestyle='dotted')
ax1.text(1, 6, '5 km', color='gray')


# plot amplitudes (ax1)
c_amp = default_color(0)
ax1.plot(amp_delD, alt / 1000, color=c_amp, linestyle='-.')
ax1.plot(amp_H2O, alt / 1000, color=c_amp, linestyle='-')
ax1.set_ylabel('altitude [km]')
ax1.set_xlabel('amplitude')
ax1.xaxis.label.set_color(c_amp)
ax1.tick_params(axis='x', colors=c_amp)

# plot standard deviation (ax2)
c_sig = default_color(1)
ax2 = ax1.twiny()
ax2.plot(sigma, alt / 1000, color=c_sig)
ax2.set_xlabel('standard deviation')
ax2.tick_params(axis='x', colors=c_sig)
ax2.xaxis.label.set_color(c_sig)
plt.show()

# %% [markdown]

# Greenhouse gases
cov = ghg.assumed_covariance(0)

plt.imshow(cov)
plt.colorbar()
plt.show()
# %% [markdown]

# Nitrid Acid
cov = hno3.assumed_covariance(0)

plt.imshow(cov)
plt.colorbar()
plt.show()

# %%
