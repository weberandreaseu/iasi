import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.pipeline import Pipeline
from analysis.data import GeographicArea
from sklearn.metrics import davies_bouldin_score

class Sink:

    def __init__(self, pipeline: Pipeline, area: GeographicArea):
        self.pipeline = pipeline
        self.area = area

    def save(self):
        raise NotImplementedError

    def satistics(self, X, y, params):
        scaler = self.pipeline.named_steps['scaler']
        X_ = scaler.fit_transform(X)
        total = len(X)
        noise_mask = y > -1
        # number of cluster excluding noise
        cluster = len(np.unique(y[noise_mask]))
        noise = total - len(y[noise_mask])
        if cluster > 1:
            db_score = davies_bouldin_score(X_[noise_mask, :], y[noise_mask])
        else:
            db_score = np.nan
        # print("{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10.5f}".format(total, self.pipeline.get_params(), noise, cluster, db_score))
        print(params)



class NetCDFSink(Sink):

    def __init__(self,
                 pipeline: Pipeline,
                 area: GeographicArea,
                 H2O_bins=15,
                 delD_bins=20):
        super(NetCDFSink, self).__init__(pipeline, area)
        self.H2O_bins = H2O_bins
        self.delD_bins = delD_bins

    def create_dataset(self, filename) -> Dataset:
        nc = Dataset(filename, 'w', format='NETCDF4')
        nc.setncattr_string('lat', self.area.lat)
        nc.setncattr_string('lon', self.area.lon)
        nc.createDimension('H20_bins', self.H2O_bins)
        nc.createDimension('delD_bins', self.delD_bins)
        # unlimted clusters
        nc.createDimension('cluster', None)
        nc.createVariable(
            'cluster', 'u4', ('cluster', 'delD_bins',  'H20_bins'))
        return nc

    def save(self, df: pd.DataFrame, filename='cluster.nc'):
        assert -1 not in df.label
        with self.create_dataset(filename) as nc:
            cluster = nc['cluster']
            for group, data in df.groupby(['label']):
                hist, xedges, yedges = np.histogram2d(
                    data['H2O'].values, data['delD'].values, bins=(self.H2O_bins, self.delD_bins))
                cluster[group] = hist.T
