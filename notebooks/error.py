# %%

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from iasi import Composition
from iasi.quadrant import AssembleFourQuadrants
# water vapour

# Traf transformiert vom {ln[H2O], ln[HDO]} Koordinatensystem ins
# {0.5*(ln[H2O]+ln[HDO]),  ln[HDO]-ln[H2O]} Koordinatensystem.

event = 0
nc = Dataset('/tmp/iasi/compression/MOTIV-single-event.nc')
# nc = Dataset('/tmp/iasi/groups/MOTIV-single-event.nc')
nol = nc['atm_nol'][...]
alt = nc['atm_altitude'][...]
avk = nc['/state/WV/atm_avk']
composition = Composition.factory(avk)
avk = composition.reconstruct(nol)
avk = AssembleFourQuadrants(None).transform(avk[event], nol.data[event])


class Error:
    traf = np.block([
        [np.diag(np.full(nol, +0.5)), np.diag(np.full(nol, 0.5))],
        [np.diag(np.full(nol, -1.0)), np.diag(np.full(nol, 1.0))]
    ])

    def __init__(self, matrix):
        self.matrix = matrix

    def gaussian(self, x, mu, sig):
        """Gaussian function

        :param x:   Input value
        :param mu:  Mean value of gaussian
        :param sig: Standard deviation of gaussian
        """
        return np.exp(-((x - mu)*(x - mu))/(2 * sig * sig))

    def get_convariance_matrix(self, alt, nol):
        s_atm_traf = np.zeros((2 * nol, 2 * nol))
        for i in range(nol):
            for j in range(nol):
                s_atm_traf[i, j] = self.gaussian(alt[i], alt[j], 2500)
                s_atm_traf[i + nol, j + nol] = 0.1 * \
                    self.gaussian(alt[i], alt[j], 2500)
        # s_atm = s_cov
        # traf * s_atm * traf = s_atm_traf
        # s_atm = inv(traf) * s_atm_traf * inv(traf)
        s_atm = np.linalg.inv(self.traf) @ \
            s_atm_traf @ \
            np.linalg.inv(self.traf)
        return (self.matrix - np.eye(nol * 2)) * s_atm * (self.matrix - np.eye(nol*2)).T


metric = Error(avk)
cov = metric.get_convariance_matrix(alt[event], nol[event])

plt.imshow(cov)
plt.colorbar()
plt.title('S err')
plt.show()


# %%
np.diag(cov)
plt.plot(np.diag(cov))
plt.show()
