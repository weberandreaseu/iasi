# %%
from iasi.metrics import Covariance
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from iasi.composition import Composition
from iasi.quadrant import AssembleFourQuadrants
# water vapour

# Traf transformiert vom {ln[H2O], ln[HDO]} Koordinatensystem ins
# {0.5*(ln[H2O]+ln[HDO]),  ln[HDO]-ln[H2O]} Koordinatensystem.

event = 0
nc_rc = Dataset('/tmp/iasi/compression/0.001/MOTIV-single-event.nc')
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

cov = Covariance(level, alt)
s_err = cov.smoothing_error_covariance(avk, np.eye(2*level))
s_diff = cov.smoothing_error_covariance(avk, avk_rc)

plt.imshow(cov.apriori_covariance())
# plt.imshow((avk - np.identity(2 * level)).T)
plt.colorbar()
plt.title('S atm')
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
