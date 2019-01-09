import glob
import logging
import time
from datetime import datetime
from typing import Dict

from netCDF4 import Dataset
import numpy as np
import pandas as pd


class DeltaDRetrieval():
    mandadory_input_variables = [
        'state_WVatm_avk',
        'state_WVatm',
        'atm_altitude',
        'state_WVatm_a',
        'atm_nol'
    ]

    output_variables = [
        'H2O', 'delD', 'lat', 'lon', 'datetime', 'fqual', 'iter',
        'srf_flag', 'srf_alt', 'dofs_T2', 'atm_alt', 'Sens'
    ]

    level_of_interest = -19  # 5km height from above

    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)

    def describe(self) -> Dict:
        n_events = 0
        for file in self.files:
            nc = Dataset(file)
            n_events += nc.dimensions['event'].size
            nc.close()
        logging.info(
            f'Number of files: {len(self.files)}, Number of events: {n_events}')
        return {'n_files': len(self.files), 'n_events': n_events}

    def retrieve(self) -> pd.DataFrame:
        data = []
        for f in self.files:
            nc = Dataset(f)
        #     try:
            events = nc.dimensions['event'].size
            level_of_interest = -19  # 5km height from above
            avk = nc['state_WVatm_avk'][...]
            wv = nc['state_WVatm'][...]
            atm_alt = nc['atm_altitude'][...]
            wv_a = nc['state_WVatm_a'][...]
            nol = nc['atm_nol'][...]
            h2o = []
            hdo = []
            dof = []
            sens = []
            alt = []
            for event in range(events):
                if isinstance(avk, np.ma.MaskedArray) and avk[event].mask.all():
                    continue
                n = int(nol[event])
                P = np.block([[np.identity(n)*0.5, np.identity(n)*0.5],
                              [-np.identity(n), np.identity(n)]])
                P_inv = np.linalg.inv(P)
                A = np.block([[avk[event, 0, 0, :n, :n], avk[event, 0, 1, :n, :n]], [
                    avk[event, 1, 0, :n, :n], avk[event, 1, 1, :n, :n]]]).T
                A_ = P @ A @ P_inv
                C = np.block([[A_[n:, n:], np.zeros((n, n))],
                              [-A_[n:, :n], np.identity(n)]])
                A__ = C @ A_
                x = C @ P @ (wv[event][:n, :n].reshape(-1) - wv_a[event]
                             [:n, :n].reshape(-1)) + P @ wv_a[event][:n, :n].reshape(-1)
                corrlength = 5000.
                S = np.exp(-((atm_alt[event, :n, np.newaxis] -
                              atm_alt[event, np.newaxis, :n])**2 / (2 * corrlength**2)))
                corrlengthbl = 500.
                Sbl = np.exp(-((atm_alt[event, :n, np.newaxis] -
                                atm_alt[event, np.newaxis, :n])**2 / (2 * corrlengthbl**2)))
                S_ = np.block([[S[:3, :3], Sbl[3:, :3].T],
                               [Sbl[3:, :3], S[3:, 3:]]])
                #S_[S_ == np.inf] = 0.0
                #S_ = S_[:n,:n]
                S__ = (A__[:n, :n] - np.identity(n)
                       ) @ S_ @ (A__[:n, :n] - np.identity(n)).T
                h2o.append(np.exp(x[level_of_interest - n] -
                                  x[level_of_interest] / 2))
                hdo.append(1000 * (np.exp(x[level_of_interest]) - 1))
                dof.append(np.trace(A__[:n, :n]))
                sens.append(np.sqrt(S__[level_of_interest, level_of_interest]))
                alt.append(atm_alt[event, level_of_interest + n])
            lat = nc['lat'][...]
            lon = nc['lon'][...]
            local_time = nc['LT_hours'][...]
            utc_time = nc['Time'][...]
            date = nc['Date'][...]
            fit_quality = nc['fit_quality'][...]
            iterations = nc['iter'][...]
            srf_flag = nc['srf_flag'][...]
            srf_alt = nc['atm_altitude'][:, 0]
            dates = []
            for d, t in zip(date.astype(int), utc_time.astype(int)):
                try:
                    dates.append("{} {:06d}".format(d, t))
                except:
                    dates.append(d)
            dates = pd.to_datetime(
                dates, format='%Y%m%d %H:%M:%S', errors='coerce')
            data.append(pd.DataFrame({
                "H2O": h2o,
                "delD": hdo,
                "lat": lat.compressed() if isinstance(lat, np.ma.MaskedArray) else lat,
                "lon": lon.compressed() if isinstance(lon, np.ma.MaskedArray) else lon,
                "datetime": dates.dropna(),
                "fqual": fit_quality.compressed() if isinstance(fit_quality, np.ma.MaskedArray) else fit_quality,
                "iter": iterations.compressed() if isinstance(iterations, np.ma.MaskedArray) else iterations,
                "srf_flag": srf_flag.compressed() if isinstance(srf_flag, np.ma.MaskedArray) else srf_flag,
                "srf_alt": srf_alt.compressed() if isinstance(srf_alt, np.ma.MaskedArray) else srf_alt,
                "dofs_T2": dof,
                # atm_alt[:,level_of_interest].compressed() if isinstance(atm_alt,np.ma.MaskedArray) else atm_alt[:,level_of_interest],
                "atm_alt": alt,
                "Sens": sens}))
        #     except Exception as e:
        #         print(time.ctime())
        #         print("ERROR: ", f, event)
        #         print(e)
        #         raise(e)

            nc.close()
        return pd.concat(data, ignore_index=True)
