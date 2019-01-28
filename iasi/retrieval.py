import glob
import logging
import os
import time
from datetime import datetime
from typing import Dict

import luigi
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from iasi.svd import SingularValueDecomposition
from iasi.file import CopyNetcdfFile, ReadFile
from iasi.util import CustomTask


class DeltaDRetrieval(CustomTask):
    # 5km height from above
    level_of_interest = luigi.IntParameter(default=-19)
    svd = luigi.BoolParameter()
    dim = luigi.IntParameter(default=14)
    file = luigi.Parameter()

    output_variables = [
        'H2O', 'delD', 'lat', 'lon', 'datetime', 'fqual', 'iter',
        'srf_flag', 'srf_alt', 'dofs_T2', 'atm_alt', 'Sens'
    ]

    calculated = ['H2O', 'delD', 'dofs_T2', 'Sens', 'atm_alt']

    def requires(self):
        if self.svd:
            return SingularValueDecomposition(dst=self.dst, file=self.file, dim=self.dim)
        return ReadFile(file=self.file)

    def output(self):
        if self.svd:
            subdirectories = ['retrieval', 'svd', str(self.dim)]
        else:
            subdirectories = ['retrieval', 'direct']
        return self.create_local_target(*subdirectories, file=self.file, ext='csv')

    def run(self) -> pd.DataFrame:
        with Dataset(self.input().path, 'r') as nc:
            events = nc.dimensions['event'].size
            self.avk = nc['state_WVatm_avk'][...] if not self.svd else self.reconstruct(
                nc)
            self.wv = nc['state_WVatm'][...]
            self.atm_alt = nc['atm_altitude'][...]
            self.wv_a = nc['state_WVatm_a'][...]
            self.nol = nc['atm_nol'][...]
            self.result = {var: [] for var in self.calculated}
        for event in range(events):
            self.process_event(event)
        df = pd.DataFrame(self.result)
        with self.output().open('w') as file:
            df.to_csv(file, index=None)

    def process_event(self, event: int):
        if isinstance(self.avk, np.ma.MaskedArray) and self.avk[event].mask.all():
            return
        n = int(self.nol[event])
        P = np.block([[np.identity(n)*0.5, np.identity(n)*0.5],
                      [-np.identity(n), np.identity(n)]])
        P_inv = np.linalg.inv(P)
        A = np.block([[self.avk[event, 0, 0, :n, :n], self.avk[event, 0, 1, :n, :n]], [
            self.avk[event, 1, 0, :n, :n], self.avk[event, 1, 1, :n, :n]]]).T
        A_ = P @ A @ P_inv
        C = np.block([[A_[n:, n:], np.zeros((n, n))],
                      [-A_[n:, :n], np.identity(n)]])
        A__ = C @ A_
        x = C @ P @ (self.wv[event][:n, :n].reshape(-1) - self.wv_a[event]
                     [:n, :n].reshape(-1)) + P @ self.wv_a[event][:n, :n].reshape(-1)
        corrlength = 5000.
        S = np.exp(-((self.atm_alt[event, :n, np.newaxis] -
                      self.atm_alt[event, np.newaxis, :n])**2 / (2 * corrlength**2)))
        corrlengthbl = 500.
        Sbl = np.exp(-((self.atm_alt[event, :n, np.newaxis] -
                        self.atm_alt[event, np.newaxis, :n])**2 / (2 * corrlengthbl**2)))
        S_ = np.block([[S[:3, :3], Sbl[3:, :3].T],
                       [Sbl[3:, :3], S[3:, 3:]]])
        #S_[S_ == np.inf] = 0.0
        #S_ = S_[:n,:n]
        S__ = (A__[:n, :n] - np.identity(n)
               ) @ S_ @ (A__[:n, :n] - np.identity(n)).T
        self.result['H2O'].append(
            np.exp(x[self.level_of_interest - n] - x[self.level_of_interest] / 2))
        self.result['delD'].append(
            1000 * (np.exp(x[self.level_of_interest]) - 1))
        self.result['dofs_T2'].append(np.trace(A__[:n, :n]))
        self.result['Sens'].append(
            np.sqrt(S__[self.level_of_interest, self.level_of_interest]))
        self.result['atm_alt'].append(
            self.atm_alt[event, self.level_of_interest + n])

    def reconstruct(self, dataset: Dataset) -> np.ndarray:
        avk_U = dataset['state_WVatm_avk_U'][...]
        avk_s = dataset['state_WVatm_avk_s'][...]
        avk_Vh = dataset['state_WVatm_avk_Vh'][...]
        events = dataset.dimensions['event'].size
        grid_levels = dataset.dimensions['atmospheric_grid_levels'].size
        species = dataset.dimensions['atmospheric_species'].size
        result = np.ndarray(
            shape=(events, species, species, grid_levels, grid_levels),
            dtype=np.float64
        )
        for event in range(events):
            for row in range(species):
                for column in range(species):
                    U = avk_U[event, row, column, :]
                    s = avk_s[event, row, column, :]
                    Vh = avk_Vh[event, row, column, :]
                    sigma = np.diag(s)
                    result[event, row, column] = np.dot(U, np.dot(sigma, Vh))
        return result

    # def retrieve(self) -> pd.DataFrame:
    #     for d, t in zip(date.astype(int), utc_time.astype(int)):
    #         try:
    #             dates.append("{} {:06d}".format(d, t))
    #         except:
    #             dates.append(d)
    #     dates = pd.to_datetime(
    #         dates, format='%Y%m%d %H:%M:%S', errors='coerce')
    #     data.append(pd.DataFrame({
    #         "H2O": h2o,
    #         "delD": hdo,
    #         "lat": lat.compressed() if isinstance(lat, np.ma.MaskedArray) else lat,
    #         "lon": lon.compressed() if isinstance(lon, np.ma.MaskedArray) else lon,
    #         "datetime": dates.dropna(),
    #         "fqual": fit_quality.compressed() if isinstance(fit_quality, np.ma.MaskedArray) else fit_quality,
    #         "iter": iterations.compressed() if isinstance(iterations, np.ma.MaskedArray) else iterations,
    #         "srf_flag": srf_flag.compressed() if isinstance(srf_flag, np.ma.MaskedArray) else srf_flag,
    #         "srf_alt": srf_alt.compressed() if isinstance(srf_alt, np.ma.MaskedArray) else srf_alt,
    #         "dofs_T2": dof,
    #         # atm_alt[:,level_of_interest].compressed() if isinstance(atm_alt,np.ma.MaskedArray) else atm_alt[:,level_of_interest],
    #         "atm_alt": alt,
    #         "Sens": sens}))

    # def describe(self) -> Dict:
    #     n_events = 0
    #     for file in self.files:
    #         nc = Dataset(file)
    #         n_events += nc.dimensions['event'].size
    #         nc.close()
    #     logging.info(
    #         f'Number of files: {len(self.files)}, Number of events: {n_events}')
    #     return {'n_files': len(self.files), 'n_events': n_events}
