"""
Example of running a clustering pipeline with spatio-temporal data
"""
from analysis.data import GeographicArea

area = GeographicArea(lat=(50, -25), lon=(-45, 60))
df = area.import_dataset('test/resources/METOPAB_20160101_global_evening_1000.nc')
# df = area.import_dataset('data/input/METOPAB_20160101_global_evening.nc')