#%% 
from iasi.metrics import Covariance
from netCDF4 import Dataset
import matplotlib.pyplot as plt

nc = Dataset('test/resources/MOTIV-single-event.nc')
nol = nc['atm_nol'][0]
alt = nc['atm_altitude'][0]
alt_tropo = nc['tropopause_altitude'][...].data[0]

cov = Covariance(nol, alt, alt_tropo)
S_a = cov._calc_Sa_(alt_tropo)