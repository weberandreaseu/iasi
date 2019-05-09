import unittest
from iasi.decomposition import Decomposition
from netCDF4 import Dataset


class TestThreshold(unittest.TestCase):

    def test_default_values(self):
        dummy = Dataset('dummy.nc', 'w', diskless=True, persist=False)
        group = dummy.createGroup('WV')
        group.createDimension('finite', 4)
        variable = group.createVariable('avk', 'f4', ('finite'))

        # if specified use custom threshold
        dec = Decomposition(group, variable, threshold=1e-6)
        self.assertEqual(dec.threshold, 1e-6)

        # if not specified use default value
        dec = Decomposition(group, variable)
        self.assertEqual(dec.threshold, 1e-3)

        # WV
        self.assertEqual(dec.default_threshold('WV', 'avk'), 1e-3)
        self.assertEqual(dec.default_threshold('WV', 'n'), 1e-3)
        self.assertEqual(dec.default_threshold('WV', 'Tatmxavk'), 1e-2)
        # GHG
        self.assertEqual(dec.default_threshold('GHG', 'avk'), 1e-3)
        self.assertEqual(dec.default_threshold('GHG', 'n'), 1e-3)
        self.assertEqual(dec.default_threshold('GHG', 'Tatmxavk'), 1e-2)
        # HNO3
        self.assertEqual(dec.default_threshold('HNO3', 'avk'), 1e-3)
        self.assertEqual(dec.default_threshold('HNO3', 'n'), 1e-4)
        self.assertEqual(dec.default_threshold('HNO3', 'Tatmxavk'), 1e-2)
        # Tatm
        self.assertEqual(dec.default_threshold('Tatm', 'avk'), 1e-2)
        self.assertEqual(dec.default_threshold('Tatm', 'n'), 1e-4)
