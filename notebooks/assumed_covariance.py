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
tatm = eval.NitridAcid('Tatm', nol, alt, alt_trop=alt_trop)

nol = nol.data[0]
alt = alt.data[0, :nol]
alt_trop = alt_trop.data[0]


def default_color(i: int):
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][i]


def plot_std_and_amp(sig, amp, text_pos=1):
    fig, ax1 = plt.subplots(figsize=(4, 4))
    # stratosphere
    ax1.axhline(25, color='gray', linestyle='dotted')
    ax1.text(text_pos, 26, 'stratosphere',  color='gray')
    # tropopause
    ax1.axhline(alt_trop / 1000, color='gray', linestyle='dotted')
    ax1.text(text_pos, (alt_trop / 1000) + 1, 'tropopause', color='gray')

    # plot amplitudes (ax1)
    c_amp = default_color(0)
    if len(amp) == 2:
        ax1.plot(amp[0], alt / 1000, color=c_amp, linestyle='-')
        ax1.plot(amp[1], alt / 1000, color=c_amp, linestyle='-.')
    else:
        ax1.plot(amp, alt / 1000, color=c_amp, linestyle='-')
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
    return ax1, ax2


# %% [markdown]
# ## Water Vapour
cov = wv.assumed_covariance(0)
amp_H2O, amp_delD, sigma = wv.assumed_covariance(0, return_tuple=True)

plt.imshow(cov[:nol, :nol])
plt.colorbar()
plt.xlabel('level')
plt.ylabel('level')
plt.show()

plt.imshow(cov[nol:, nol:])
plt.colorbar()
plt.xlabel('level')
plt.ylabel('level')
plt.show()

text_pos = 0.8
ax1, ax2 = plot_std_and_amp(sigma, [amp_H2O, amp_delD], text_pos=text_pos)
# 5km
ax1.axhline(5, color='gray', linestyle='dotted')
ax1.text(text_pos, 6, '5 km', color='gray')
plt.show()


# %% [markdown]

# ## Greenhouse gases
cov = ghg.assumed_covariance(0)
plt.imshow(cov[:nol, :nol])
plt.colorbar()
plt.xlabel('level')
plt.ylabel('level')
plt.show()

sig = ghg.sigma(0, 0.6)
amp = ghg._amplitude(0)

plot_std_and_amp(sig, amp, text_pos=0.18)
plt.show()

# %% [markdown]

# ## Nitrid Acid
cov = hno3.assumed_covariance(0)
plt.imshow(cov)
plt.colorbar()
plt.show()

sig = hno3.sigma(0, f_sigma=1.2)
amp = hno3._amplitude(0)

text_pos = 1.4
ax1, ax2 = plot_std_and_amp(sig, amp, text_pos=text_pos)

tropo_4 = (alt_trop - 4000) / 1000
ax1.axhline(tropo_4, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_4 + 1, 'tropopause - 4 km', color='gray')

tropo_8 = (alt_trop + 8000) / 1000
ax1.axhline(tropo_8, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_8 + 1, 'tropopause + 8 km', color='gray')

plt.show()

# %% [markdown]

# ## Temperature

cov = tatm.assumed_covariance_temperature(0)
plt.imshow(cov)
plt.colorbar()
plt.show()

text_pos=1.5
amp, sig = tatm.assumed_covariance_temperature(0, return_tuple=True)
ax1, ax2 = plot_std_and_amp(sig, amp, text_pos=text_pos)

srf_4 = (alt[0] + 4000) / 1000
ax1.axhline(srf_4, color='gray', linestyle='dotted')
ax1.text(text_pos, srf_4 + 1, 'surface + 4 km', color='gray')

tropo_5 = (alt_trop + 5000) / 1000
ax1.axhline(tropo_5, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_5 + 1, 'tropopause + 5 km', color='gray')

plt.show()