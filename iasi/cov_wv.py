# %%
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from functools import partial

nc = Dataset('test/resources/MOTIV-single-event.nc')
nol = 28

alt = nc['atm_altitude'][0, :nol].data
# altitude tropopause
alt_trop = 9486.431
# alt_trop = 3800
# altitude stratosphere TODO end of stratosphere?
alt_strat = 25000


def _get_s_par_wv(alt, alt0, alt_trop, alt_strat=25000, f_sigma=1.):

    if alt < 5000.:
        amp_H2O = 0.75 * (1 + alt / 5000)
        amp_dD = 0.09 * (1 + alt / 5000)
        sigma = f_sigma * 1500. * (1. + (alt - alt0) / (alt_trop - alt0))
    elif alt >= 5000. and alt < alt_trop:
        amp_H2O = 1.5
        amp_dD = 0.18
        sigma = f_sigma * 1500. * (1. + (alt - alt0) / (alt_trop - alt0))
    elif alt >= alt_trop and alt < alt_strat:
        amp_H2O = 1.5 - 1.2 * (alt - alt_trop) / (alt_strat - alt_trop)
        amp_dD = 0.18 - 0.12 * (alt - alt_trop) / (alt_strat - alt_trop)
        sigma = f_sigma * 3000. * \
            (1. + (alt - alt_trop) / (alt_strat - alt_trop))
    elif alt >= alt_strat:
        amp_H2O = 0.3
        amp_dD = 0.06
        sigma = f_sigma * 6000.
    else:
        raise ValueError('Invalid altitude')

    return amp_H2O, amp_dD, sigma


def _get_s_par_t(alt, alt0, alt_trop):
    if alt0+4000 < alt_trop:
        # setting amp_T
        if alt <= alt0+4000:
            amp_T = 2.0 - 1.0 * (alt - alt0) / 4000
        elif alt >= alt0+4000 and alt <= alt_trop:
            amp_T = 1.
        elif alt > alt_trop and alt <= alt_trop+5000:
            amp_T = 1.0 + 0.5 * (alt - alt_trop) / 5000
        elif alt > alt_trop+5000:
            amp_T = 1.5

        # setting sigmaT
        if alt < alt_trop:
            sigmaT = 2500 * (1 + (alt - alt0) / (alt_trop - alt0))
        elif alt >= alt_trop and alt < alt_trop+10000:
            sigmaT = 5000 * (1 + (alt - alt_trop) / 10000)
        elif alt >= alt_trop+10000:
            sigmaT = 10000
    else:
        # setting amp_T
        if alt < alt_trop:
            amp_T = 2.0 - 1.0 * (alt - alt0) / (alt_trop - alt0)
        elif alt == alt_trop:
            amp_T = 1.
        elif alt > alt_trop and alt <= alt_trop+5000:
            amp_T = 1.0 + 0.5 * (alt - alt_trop) / 5000
        elif alt > alt_trop+5000:
            amp_T = 1.5

        # setting sigmaT
        if alt < alt_trop:
            sigmaT = 2500 * (1 + (alt - alt0) / (alt_trop - alt0))
        elif alt >= alt_trop and alt < alt_trop+10000:
            sigmaT = 5000 * (1 + (alt - alt_trop) / 10000)
        elif alt >= alt_trop+10000:
            sigmaT = 10000

    sigmaT = sigmaT * 3/5  # 0.2

    return amp_T, sigmaT


def _calc_Sa_(alt_trop, alt_strat=25000, f_sigma=1.):
    amp_H2O = np.zeros(nol)  # , dtype='float64')
    amp_dD = np.zeros(nol)  # , dtype='float64')
    sigma = np.zeros(nol)  # , dtype='float64')
    fct_partial = partial(_get_s_par_wv, alt0=alt[0],
                          alt_trop=alt_trop, alt_strat=alt_strat, f_sigma=f_sigma)
    results = map(fct_partial, alt)

    for ires, res in enumerate(results):
        amp_H2O[ires] = res[0]
        amp_dD[ires] = res[1]
        sigma[ires] = res[2]

    S_H2O = amp_H2O[:, np.newaxis] * amp_H2O[np.newaxis, :] \
        * np.exp(-(alt[:, np.newaxis] - alt[np.newaxis, :])**2 / (2 * sigma[:, np.newaxis] * sigma[np.newaxis, :]))
    S_dD = amp_dD[:, np.newaxis] * amp_dD[np.newaxis, :] \
        * np.exp(-(alt[:, np.newaxis] - alt[np.newaxis, :])**2 / (2 * sigma[:, np.newaxis] * sigma[np.newaxis, :]))

    S_H2O = np.asarray(S_H2O)  # , dtype='float32')
    S_dD = np.asarray(S_dD)  # ,  dtype='float32')

    Sa_ = np.zeros([2*nol, 2*nol])  # , dtype='float32')
    Sa_[:nol, :nol] = S_H2O
    Sa_[nol:, nol:] = S_dD
    return Sa_


def _calc_SaT():

    amp_T = np.zeros(nol)
    sigma_T = np.zeros(nol)

    fct_partial = partial(
        _get_s_par_t, alt0=alt[0], alt_trop=alt_trop)
    results = map(fct_partial, alt)

    for ires, res in enumerate(results):
        amp_T[ires] = res[0]
        sigma_T[ires] = res[1]

    SaT = amp_T[:, np.newaxis] * amp_T[np.newaxis, :] \
        * np.exp(-(alt[:, np.newaxis] - alt[np.newaxis])**2 / (2 * sigma_T[:, np.newaxis] * sigma_T[np.newaxis, :]))
    SaT = np.asarray(SaT.T)  # , dtype='float32')

    return SaT


#%%
sa = _calc_Sa_(alt_trop)
plt.imshow(sa)