from math import cos, pi

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 1 degree lat ~ 111 km
latitude_km = 111.0

class SpatialWaterVapourScaler(BaseEstimator, TransformerMixin):
    """
    Scale water vapour features including latitude and longitude
    """

    def __init__(self, delD=10, H2O=0.1, km=60):
        self.delD = delD
        self.H2O = H2O
        self.km = km

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Scale 4 water vapour features according to transform parameters

        order of colums:
        - lat
        - lon
        - H2O
        - delD
        """
        assert X.shape[1] == 4
        # lon
        # cos takes radians => pi radians are 180 degrees
        # lon = lon * latitude_km * cos(lat * (pi / 180))
        X[:, 1] = (X[:, 1] * latitude_km *
                   np.cos(X[:, 0] * (pi / 180))) / self.km
        # lat
        X[:, 0] = (X[:, 0] * latitude_km) / self.km
        # H2O
        X[:, 2] = np.log(X[:, 2]) / self.H2O
        # delD
        X[:, 3] = X[:, 3] / self.delD
        return X
