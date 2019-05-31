from functools import partial

import numpy as np


class Covariance:
    def __init__(self, nol: int, alt: np.ma.MaskedArray, alt_tropo: np.ma.MaskedArray):
        """assumed covariances

        :param no           number of levels
        :param alt          altitudes
        :param alt_tropo    tropopause_altitude
        """

        self.nol = nol
        self.alt = alt
        self.alt_tropo = alt_tropo

    def gaussian(self, x, mu, sig):
        """Gaussian function

        :param x:   Input value
        :param mu:  Mean value of gaussian
        :param sig: Standard deviation of gaussian
        """
        return np.exp(-((x - mu)*(x - mu))/(2 * sig * sig))

    def traf(self) -> np.ndarray:
        """P (see equation 6)

        Used to transform {ln[H2O], ln[HDO]} state
        into the new coordination systems
        {(ln[H2O]+ln[HDO])/2 and ln[HDO]-ln[H2O]} 
        """
        return np.block([[np.identity(self.nol)*0.5, np.identity(self.nol)*0.5],
                         [-np.identity(self.nol), np.identity(self.nol)]])

    def assumed_covariance(self, species=2, w1=1.0, w2=0.01, correlation_length=2500) -> np.ndarray:
        """Sa' (see equation 7)

        A priori covariance of {(ln[H2O]+ln[HDO])/2 and ln[HDO]-ln[H2O]} state
        Sa See equation 5 in paper

        :param species              Number of atmospheric species (1 or 2)
        :param w1:                  Weight for upper left quadrant
        :param w2:                  Weight for lower right quadrant (ignored with 1 species)
        :param correlation_length:  Assumed correlation of atmospheric levels in meter
        """
        # only 1 or 2 species are allowed
        assert (species >= 1) and (species <= 2)
        result = np.zeros((species * self.nol, species * self.nol))
        for i in range(self.nol):
            for j in range(self.nol):
                # 2500 = correlation length
                # 100% for
                # (ln[H2O]+ln[HDO])/2 state
                result[i, j] = w1 * \
                    self.gaussian(self.alt[i], self.alt[j], correlation_length)
                if species == 2:
                    # 10% for (0.01 covariance)
                    # ln[HDO]-ln[H2O] state
                    result[i + self.nol, j + self.nol] = w2 * \
                        self.gaussian(
                            self.alt[i], self.alt[j], correlation_length)
        return result

    def apriori_covariance(self) -> np.ndarray:
        """Sa (see equation 5)

        A priori Covariance of {ln[H2O], ln[HDO]} state

        Sa' = P * Sa * P.T (equation 7 in paper)
        equals to 
        Sa = inv(P) * Sa' * inv(P.T)
        """
        P = self.traf()
        return np.linalg.inv(P) @ self.apriori_covariance_traf() @ np.linalg.inv(P.T)

    def type1_of(self, matrix) -> np.ndarray:
        """A' (see equation 10)

        Return tranformed martix
        """
        P = self.traf()
        return P @ matrix @ np.linalg.inv(P)

    def c_by_type1(self, A_) -> np.ndarray:
        return np.block([[A_[self.nol:, self.nol:], np.zeros((self.nol, self.nol))],
                         [-A_[self.nol:, :self.nol], np.identity(self.nol)]])

    def c_by_avk(self, avk):
        A_ = self.type1_of(avk)
        return self.c_by_type1(A_)

    def type2_of(self, matrix) -> np.ndarray:
        """A'' (see equation 15)

        A posteriori transformed matrix 
        """
        A_ = self.type1_of(matrix)
        C = self.c_by_type1(A_)
        return C @ A_

    def smoothing_error(self, actual_matrix, to_compare, **kwargs) -> np.ndarray:
        """S's (see equation 11)
        """
        return (actual_matrix - to_compare) @ self.assumed_covariance(**kwargs) @ (actual_matrix - to_compare).T

    # TODO refactor 
    def _get_s_par_wv(self, alt, alt0, alt_trop, alt_strat=25000, f_sigma=1.):

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

    def _get_s_par_t(self, alt, alt0, alt_trop):
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

    def _calc_Sa_(self, alt_trop, alt_strat=25000, f_sigma=1., return_amp=False):
        amp_H2O = np.zeros(self.nol)  # , dtype='float64')
        amp_dD = np.zeros(self.nol)  # , dtype='float64')
        sigma = np.zeros(self.nol)  # , dtype='float64')
        fct_partial = partial(self._get_s_par_wv, alt0=self.alt[0],
                              alt_trop=alt_trop, alt_strat=alt_strat, f_sigma=f_sigma)
        results = map(fct_partial, self.alt)

        for ires, res in enumerate(results):
            amp_H2O[ires] = res[0]
            amp_dD[ires] = res[1]
            sigma[ires] = res[2]

        S_H2O = amp_H2O[:, np.newaxis] * amp_H2O[np.newaxis, :] \
            * np.exp(-(self.alt[:, np.newaxis] - self.alt[np.newaxis, :])**2 / (2 * sigma[:, np.newaxis] * sigma[np.newaxis, :]))
        S_dD = amp_dD[:, np.newaxis] * amp_dD[np.newaxis, :] \
            * np.exp(-(self.alt[:, np.newaxis] - self.alt[np.newaxis, :])**2 / (2 * sigma[:, np.newaxis] * sigma[np.newaxis, :]))

        S_H2O = np.asarray(S_H2O)  # , dtype='float32')
        S_dD = np.asarray(S_dD)  # ,  dtype='float32')

        Sa_ = np.zeros([2*self.nol, 2*self.nol])  # , dtype='float32')
        Sa_[:self.nol, :self.nol] = S_H2O
        Sa_[self.nol:, self.nol:] = S_dD

        if return_amp:
            ### TEST ###
            return Sa_, amp_H2O, amp_dD, sigma
        else:
            return Sa_

    def _calc_SaT(self):

        amp_T = np.zeros(self.nol)
        sigma_T = np.zeros(self.nol)

        fct_partial = partial(self._get_s_par_t, alt0=self.alt[0], alt_trop=self.alt_trop)
        results = map(fct_partial, self.alt)

        for ires, res in enumerate(results):
            amp_T[ires] = res[0]
            sigma_T[ires] = res[1]

        SaT = amp_T[:, np.newaxis] * amp_T[np.newaxis, :] \
            * np.exp(-(self.alt[:, np.newaxis] - self.alt[np.newaxis])**2 / (2 * sigma_T[:, np.newaxis] * sigma_T[np.newaxis, :]))
        SaT = np.asarray(SaT.T)  # , dtype='float32')

        return SaT
