# %%[markdown]
# # NetCDF Compression Performance
#
# Matrices containing significant amount of fill values.
# To reduce size, we evaluate compression in netCDF4
#

import numpy as np
from timeit import default_timer as timer
from netCDF4 import Dataset
import pandas as pd
import os
import tempfile

dimensions = (10000, 28, 28)


def create_file(file: str = None, zlib=False):
    if not os.path.exists('/tmp/compression'):
        os.mkdir('/tmp/compression')
    path = os.path.join('/tmp/compression', file)
    nc = Dataset(path, 'w', format='NETCDF4')
    nc.createDimension('event', dimensions[0])
    nc.createDimension('grid_level', dimensions[1])
    nc.createVariable('dummy', 'f8',
                      ('event', 'grid_level', 'grid_level'),
                      zlib=zlib)
    return path, nc


def write_values(nc: Dataset, array: np.ndarray):
    slices = tuple(map(slice, array.shape))
    nc['dummy'][slices] = array


# sequential read is an bad idea combined with compression
# each read triggers decompression of complete variable
def sequential_read(nc: Dataset):
    events = nc.dimensions['event'].size
    for event in range(events):
        nc['dummy'][event]


samples = [
    {'file': 'full_plain.nc', 'zlib': False, 'dim': (10000, 28, 28)},
    {'file': 'full_zlib.nc', 'zlib': True, 'dim': (10000, 28, 28)},
    {'file': 'half_plain.nc', 'zlib': False, 'dim': (10000, 14, 28)},
    {'file': 'half_zlib.nc', 'zlib': True, 'dim': (10000, 14, 28)}
]

for arg in samples:
    file, nc = create_file(arg['file'], arg['zlib'])
    values = np.random.uniform(size=arg['dim'])
    t1 = timer()
    write_values(nc, values)
    t2 = timer()
    # read values
    nc['dummy'][...]
    t3 = timer()
    # sequential_read(nc)
    t4 = timer()
    arg['write_dur'] = t2 - t1
    arg['read_full'] = t3 - t2
    arg['read_seq'] = t4 - t3
    arg['size'] = os.path.getsize(file)
    nc.close()

result = pd.DataFrame(samples)
print(result)
