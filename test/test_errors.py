import unittest
from netCDF4 import Dataset
from iasi.decomposition import SingularValueDecomposition
class TestErrors:
    """Common errors that occur for some events"""

    def test_svd_converges(self):
        nc = Dataset('test/resources/METOPB_20160709011752_19753_20190314031224.nc')
        nol = 18
        group = nc['/state/WV']
        variable = nc['/state/WV/avk']
        svd = SingularValueDecomposition()