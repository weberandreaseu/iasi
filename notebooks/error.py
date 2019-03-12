# %%
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from iasi.composition import Composition
from iasi.quadrant import AssembleFourQuadrants
# water vapour

# Traf transformiert vom {ln[H2O], ln[HDO]} Koordinatensystem ins
# {0.5*(ln[H2O]+ln[HDO]),  ln[HDO]-ln[H2O]} Koordinatensystem.

event = 0
nc_rc = Dataset('/tmp/iasi/compression/MOTIV-single-event.nc')
nc = Dataset('/tmp/iasi/groups/MOTIV-single-event.nc')
nol = nc['atm_nol'][...]
alt = nc['atm_altitude'][...]
avk = nc['/state/WV/atm_avk']

composition = Composition.factory(nc_rc['/state/WV/atm_avk'])
avk_rc = composition.reconstruct(nol)
avk_rc = AssembleFourQuadrants(None).transform(avk_rc[event], nol.data[event])
avk = AssembleFourQuadrants(None).transform(avk[event], nol.data[event])

level = nol.data[event]
alt = alt.data[event]
# class Error:
traf = np.block([
    [np.diag(np.full(level, +0.5)), np.diag(np.full(level, 0.5))],
    [np.diag(np.full(level, -1.0)), np.diag(np.full(level, 1.0))]
])

def gaussian(x, mu, sig):
    """Gaussian function

    :param x:   Input value
    :param mu:  Mean value of gaussian
    :param sig: Standard deviation of gaussian
    """
    return np.exp(-((x - mu)*(x - mu))/(2 * sig * sig))


s_atm_traf = np.zeros((2 * level, 2 * level))
for i in range(level):
    for j in range(level):
        s_atm_traf[i, j] = gaussian(alt[i], alt[j], 2500)
        s_atm_traf[i + level, j + level] = 0.01 * \
            gaussian(alt[i], alt[j], 2500)

# s_atm = s_cov
# traf * s_atm * traf = s_atm_traf
# s_atm = inv(traf) * s_atm_traf * inv(traf)
s_atm = np.linalg.inv(traf) @ s_atm_traf @ np.linalg.inv(traf.T)
s_err = (avk - np.eye(level * 2)) @ s_atm @ (avk - np.eye(level * 2)).T
s_diff = (avk - avk_rc) @ s_atm @ (avk - avk_rc).T


plt.imshow(avk)
plt.colorbar()
plt.title('Avk')
plt.show()

plt.imshow(s_err)
plt.colorbar()
plt.title('S err')
plt.show()

plt.imshow(s_diff)
plt.colorbar()
plt.title('S diff')
plt.show()


np.diag(s_err)
plt.plot(np.diag(s_err))
plt.title('S err')
plt.show()

np.diag(s_diff)
plt.plot(np.diag(s_diff))
plt.title('S diff')
plt.show()


