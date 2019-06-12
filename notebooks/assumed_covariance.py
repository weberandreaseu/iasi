# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset

import iasi.evaluation as eval

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


def project_alt_to_index(alt_ref) -> float:
    alt_prev = alt[0]
    for i, alt_i in enumerate(alt):
        if alt_i > alt_ref:
            return i + (alt_ref - alt_prev) / (alt_i - alt_prev) - 1
        else:
            alt_prev = alt_i


def plot_covariance(s_cov):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    im = ax.imshow(s_cov)
    plt.ylim(-0.5, nol-0.5)
    plt.xlabel('altitude [level]')
    plt.ylabel('altitude [level]')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_amplitude(*amps, text_pos=1, alt_strat=25000):
    fig, ax1 = plt.subplots(figsize=(4, 4))
    # stratosphere
    alt_s = project_alt_to_index(alt_strat)
    ax1.axhline(alt_s, color='gray', linestyle='dotted')
    ax1.text(text_pos, alt_s + 0.6, 'stratosphere',  color='gray')
    # tropopause
    alt_t = project_alt_to_index(alt_trop)
    ax1.axhline(alt_t, color='gray', linestyle='dotted')
    ax1.text(text_pos, (alt_t) + 0.6, 'tropopause', color='gray')

    # plot amplitudes (ax1)

    for amp in amps:
        ax1.plot(amp, np.arange(nol))

    # c_amp = default_color(0)
    # if len(amp) == 2:
    #     ax1.plot(amp[0], np.arange(nol), color=c_amp, linestyle='-')
    #     # scale amp delD by to for better visualization
    #     ax1.plot(amp[1], np.arange(nol), color=c_amp, linestyle='-.')
    # else:
    #     ax1.plot(amp, np.arange(nol), color=c_amp, linestyle='-')
    ax1.set_ylabel('altitude [level]')
    ax1.set_xlabel('standard deviation [ln(ppmv)]')
    # ax1.xaxis.label.set_color(c_amp)
    # ax1.tick_params(axis='x', colors=c_amp)

    # # plot standard deviation (ax2)
    # c_sig = default_color(1)
    # ax2 = ax1.twiny()
    # ax2.plot(sigma / 1000, np.arange(nol), color=c_sig)
    # ax2.set_xlabel('standard deviation [km]')
    # ax2.tick_params(axis='x', colors=c_sig)
    # ax2.xaxis.label.set_color(c_sig)
    return ax1


# %% [markdown]
# ## Correlation length
# Same for all gases and atmosphere

def plot_correlation_lenght(sig, alt_strat=25000, text_pos=1.4):
    fig, ax1 = plt.subplots(figsize=(4, 4))
    # stratosphere
    alt_s = project_alt_to_index(alt_strat)
    ax1.axhline(alt_s, color='gray', linestyle='dotted')
    ax1.text(text_pos, alt_s + 0.6, 'stratosphere',  color='gray')
    # tropopause
    alt_t = project_alt_to_index(alt_trop)
    ax1.axhline(alt_t, color='gray', linestyle='dotted')
    ax1.text(text_pos, (alt_t) + 0.6, 'tropopause', color='gray')
    plt.plot(sig / 1000, np.arange(nol))
    plt.ylabel('altitude [level]')
    plt.xlabel('correlation length [km]')

sig = wv.sigma(0)
plot_correlation_lenght(sig)

# %% [markdown]
# ## Water Vapour
cov = wv.assumed_covariance(0)
amp_H2O, amp_delD = wv.amplitude(0)
sig = wv.sigma(0)

plot_covariance(cov[:nol, :nol])
plt.savefig('wv_cov_H2O.pdf')
plt.show()

plot_covariance(cov[nol:, nol:])
plt.savefig('wv_cov_HDO.pdf')
plt.show()

text_pos = 0.8
ax1 = plot_amplitude(amp_H2O, amp_delD, text_pos=text_pos)
# 5km
alt_5 = project_alt_to_index(5000)
ax1.axhline(alt_5, color='gray', linestyle='dotted')
ax1.text(text_pos, alt_5 + 0.6, '5 km', color='gray')
plt.savefig('wv_amp_sig.pdf', bbox_inches='tight')
plt.show()


# %% [markdown]

# ## Greenhouse gases
cov = ghg.assumed_covariance(0)
plot_covariance(cov[:nol, :nol])
plt.savefig('ghg_cov.pdf')
plt.show()

sig = ghg.sigma(0)
amp = ghg.amplitude(0)

plot_amplitude(amp, text_pos=0.18)
plt.savefig('ghg_amp_sig.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]

# ## Nitrid Acid
cov = hno3.assumed_covariance(0)
plot_covariance(cov)
plt.savefig('hno3_cov.pdf')
plt.show()

sig = hno3.sigma(0)
amp = hno3.amplitude(0)

text_pos = 1.3
ax1 = plot_amplitude(amp, text_pos=text_pos)

tropo_4 = project_alt_to_index(alt_trop - 4000)
ax1.axhline(tropo_4, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_4 + 0.6, 'tropopause - 4 km', color='gray')

tropo_8 = project_alt_to_index(alt_trop + 8000)
ax1.axhline(tropo_8, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_8 + 0.6, 'tropopause + 8 km', color='gray')

plt.savefig('hno3_amp_sig.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]

# ## Temperature

cov = tatm.assumed_covariance_temperature(0)
plot_covariance(cov)
plt.savefig('tatm_cov.pdf')
plt.show()

text_pos = 1.55
amp = tatm.amplitude_temperature(0)
sig = tatm.sigma(0)
ax1 = plot_amplitude(amp, text_pos=text_pos)

srf_4 = project_alt_to_index(alt[0] + 4000)
ax1.axhline(srf_4, color='gray', linestyle='dotted')
ax1.text(text_pos, srf_4 + 0.6, 'surface + 4 km', color='gray')

tropo_5 = project_alt_to_index(alt_trop + 5000)
ax1.axhline(tropo_5, color='gray', linestyle='dotted')
ax1.text(text_pos, tropo_5 + 0.6, 'tropopause + 5 km', color='gray')

plt.savefig('tatm_amp_sig.pdf', bbox_inches='tight')

plt.show()
