import logging

import numpy as np
from netCDF4 import Dataset

from iasi import svd


def _setup():
    np.set_printoptions(precision=4)
    logging.basicConfig(level=logging.DEBUG)


def convert(b, reduction):
    U, s, Vh = svd.decompose(b, reduction_factor=reduction)
    return svd.reconstruct(U, s, Vh)


def mse(a, b) -> float:
    return np.square(a - b).mean()


def mean_error(a, b):
    return np.abs(a - b).mean()


_setup()
m, n = 28, 28
a = np.random.rand(m, n)

for factor in np.linspace(1, 0.2, 10):
    a_ = convert(a, factor)
    logging.info(
        f'Reduction factor: {factor:.4f}, Mean error: {mean_error(a, a_):.4f}')


# def get_event():
#     nc = Dataset(
#         'data/IASI-A_20160627_50269_v2018_fast_part0_20180413215350.nc', 'r')
#     avk = nc.variables['state_WVatm_avk']
#     # get average kernel from first event
#     event = avk[0]
#     nc.close()
#     return event

# event = get_event()
# event = np.ma.asarray(event)
# event_ = convert(event, 0.5)
# print('Event Reduction factor: {}, MSE: {}'.format(0.5, mse(event, event_)))
