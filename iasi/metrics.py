import numpy as np


class Covariance:
    def __init__(self, nol: int, alt: np.ma.MaskedArray):
        self.nol = nol
        self.alt = alt

    def gaussian(self, x, mu, sig):
        """Gaussian function

        :param x:   Input value
        :param mu:  Mean value of gaussian
        :param sig: Standard deviation of gaussian
        """
        return np.exp(-((x - mu)*(x - mu))/(2 * sig * sig))

    def traf(self):
        return np.block([[np.identity(self.nol)*0.5, np.identity(self.nol)*0.5],
                         [-np.identity(self.nol), np.identity(self.nol)]])

    def s_atm_traf(self):
        result = np.zeros((2 * self.nol, 2 * self.nol))
        for i in range(self.nol):
            for j in range(self.nol):
                result[i, j] = self.gaussian(self.alt[i], self.alt[j], 2500)
                result[i + self.nol, j + self.nol] = 0.01 * \
                    self.gaussian(self.alt[i], self.alt[j], 2500)
        return result

    def s_atm(self):
        traf = self.traf()
        return np.linalg.inv(traf) @ self.s_atm_traf() @ np.linalg.inv(traf.T)

    def error(self, original, compare):
        return (original - compare) @ self.s_atm() @ (original - compare).T
