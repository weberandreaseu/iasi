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

    def traf(self) -> np.ndarray:
        """P (see equation 6)

        Used to transform {ln[H2O], ln[HDO]} state
        into the new coordination systems
        {(ln[H2O]+ln[HDO])/2 and ln[HDO]-ln[H2O]} 
        """
        return np.block([[np.identity(self.nol)*0.5, np.identity(self.nol)*0.5],
                         [-np.identity(self.nol), np.identity(self.nol)]])

    def type1_covariance(self) -> np.ndarray:
        """Sa' (see equation 7)

        A priori covariance of {(ln[H2O]+ln[HDO])/2 and ln[HDO]-ln[H2O]} state
        Sa See equation 5 in paper
        """
        result = np.zeros((2 * self.nol, 2 * self.nol))
        for i in range(self.nol):
            for j in range(self.nol):
                # 2500 = correlation length
                # 100% for
                # (ln[H2O]+ln[HDO])/2 state
                result[i, j] = self.gaussian(self.alt[i], self.alt[j], 2500)
                # 10% for (0.01 covariance)
                # ln[HDO]-ln[H2O] state
                result[i + self.nol, j + self.nol] = 0.01 * \
                    self.gaussian(self.alt[i], self.alt[j], 2500)
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

    def type2_of(self, matrix) -> np.ndarray:
        """A'' (see equation 15)

        A posteriori transformed matrix 
        """
        A_ = self.type1_of(matrix)
        C = self.c_by_type1(A_)
        return C @ A_

    def smoothing_error(self, actual_matrix, to_compare) -> np.ndarray:
        """S's (see equation 11)
        """
        return (actual_matrix - to_compare) @ self.type1_covariance() @ (actual_matrix - to_compare).T
