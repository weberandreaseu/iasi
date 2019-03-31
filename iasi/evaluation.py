import os

import luigi
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from netCDF4 import Dataset, Group, Variable

from iasi.compression import CompressDataset, SelectSingleVariable
from iasi.composition import Composition
from iasi.file import MoveVariables
from iasi.util import CustomTask
from iasi.metrics import Covariance
from iasi.quadrant import AssembleFourQuadrants


class EvaluationTask(CustomTask):
    file = luigi.Parameter()
    gases = luigi.ListParameter()
    variables = luigi.ListParameter()

    def requires(self):
        compression_parameter = {
            'compressed': [True],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'threshold': [1e-2, 1e-3, 1e-4, 1e-5],
            'gas': self.gases,
            'variable': self.variables
        }
        compressed_param_grid = list(ParameterGrid(compression_parameter))
        tasks = [SelectSingleVariable(**params)
                 for params in compressed_param_grid]
        # for uncompressed dataset we do not need multiple threshold values
        uncompressed_parameter = {
            'compressed': [False],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'threshold': [np.nan],
            'gas': self.gases,
            'variable': self.variables
        }
        uncompressed_param_grid = list(ParameterGrid(uncompressed_parameter))
        return tasks + [SelectSingleVariable(**params) for params in uncompressed_param_grid]

    def run(self):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError


class EvaluationCompressionSize(EvaluationTask):

    def output(self):
        return self.create_local_target('compression-summary', file=self.file, ext='csv')

    def size_in_kb(self, file):
        return int(os.path.getsize(file) / (1000))

    def run(self):
        # get size for all parameters
        df = pd.DataFrame()
        for task, input in zip(self.requires(), self.input()):
            df = df.append({
                'variable': task.variable,
                'compressed': task.compressed,
                'size': self.size_in_kb(input.path),
                'threshold': task.threshold
            }, ignore_index=True)
        df.to_csv(self.output().path, index=False)


class EvaluationErrorEstimation(EvaluationTask):
    def run(self):
        pass

    def output(self):
        # one error estimation report vor each gas
        return {'WV': self.create_local_target('error-estimation', 'WV', file=self.file, ext='csv')}
        # return {gas: self.create_local_target('error-estimation', gas, file=self.file, ext='csv') for gas in self.gases}

    # def run(self):
    #     df = pd.DataFrame()
    #     original = Dataset(self.input()['original'].path)
    #     nol = original['atm_nol'][...]
    #     alt = original['atm_altitude'][...]
    #     avk = original['state/WV/atm_avk'][...]
    #     original.close()
    #     wv_error = WaterVapourError(nol, alt)
    #     error = wv_error.smoothing_error_all_measurements(avk)
    #     a_error = wv_error.smoothing_error_aposteriori(avk)
    #     for task, input in zip(self.requires()['single_variable'], self.input()['single_variable']):
    #         nc = Dataset(input.path)
    #         if isinstance(nc['state/WV/atm_avk'], Variable):
    #             # avk is not decomposed
    #             avk_rc = nc['state/WV/atm_avk']
    #         else:
    #             avk_rc = Composition.factory(
    #                 nc['state/WV/atm_avk']).reconstruct(nol)
    #         diff = wv_error.smoothing_error_all_measurements(
    #             avk, avk_rc=avk_rc)
    #         a_diff = wv_error.smoothing_error_aposteriori(
    #             avk, avk_rc=avk_rc)
    #         df = df.append({
    #             'variable': task.variable,
    #             'compressed': task.compressed,
    #             'size': self.size_in_kb(input.path),
    #             'threshold': task.threshold,
    #             'err_min': error[0],
    #             'err_mean': error[1],
    #             'err_max': error[2],
    #             'err_apost_min': a_error[0],
    #             'err_apost_mean': a_error[1],
    #             'err_apost_max': a_error[2],
    #             'diff_min': diff[0],
    #             'diff_mean': diff[1],
    #             'diff_max': diff[2],
    #             'diff_apost_min': a_diff[0],
    #             'diff_apost_mean': a_diff[1],
    #             'diff_apost_max': a_diff[2]
    #             # 'file': task.file,
    #         }, ignore_index=True)
    #         nc.close()
    #     print('\n', df)
    #     df.to_csv(self.output().path, index=False)


class WaterVapourError:

    def __init__(self, nol, alt):
        self.nol = nol
        self.alt = alt

    def smoothing_error_all_measurements(self, avk, avk_rc=None, level_of_interest=-19):
        err_max = -np.inf
        err_min = np.inf
        err_mean = 0
        n = 0
        for event in range(self.nol.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            l = self.nol.data[event]
            current_level = l + level_of_interest
            if current_level < 2:
                continue
            # if reconstructed kernel not given create identity matrix and calculate covariance error
            # else use reconstructed kernel to calculate covariance difference
            cov = Covariance(l, self.alt[event])
            if avk_rc is None:
                expected = np.identity(2*l)
            else:
                expected = AssembleFourQuadrants().transform(avk_rc[event], l)
                expected = cov.avk_traf(expected)
            avk_event = AssembleFourQuadrants().transform(avk[event], l)
            a_ = cov.avk_traf(avk_event)
            s_err = cov.smoothing_error_covariance(a_, expected)
            err_max = max(err_max, s_err[current_level, current_level])
            err_min = min(err_min, s_err[current_level, current_level])
            err_mean += s_err[current_level, current_level]
            n += 1
        return (err_min, err_mean / n, err_max)

    # TODO refactor
    def smoothing_error_aposteriori(self, avk, avk_rc=None, level_of_interest=-19):
        err_max = -np.inf
        err_min = np.inf
        err_mean = 0
        n = 0
        for event in range(self.nol.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            l = self.nol.data[event]
            current_level = l + level_of_interest
            if current_level < 2:
                continue
            # if reconstructed kernel not given create identity matrix and calculate covariance error
            # else use reconstructed kernel to calculate covariance difference
            cov = Covariance(l, self.alt[event])
            if avk_rc is None:
                Arc__ = np.identity(2*l)
            else:
                avk_rc_event = AssembleFourQuadrants(
                ).transform(avk_rc[event], l)
                # A''rc
                Arc__ = cov.posteriori_traf(avk_rc_event)
            avk_event = AssembleFourQuadrants().transform(avk[event], l)
            A__ = cov.posteriori_traf(avk_event)
            s_err = cov.smoothing_error_covariance(A__, Arc__)
            err_max = max(err_max, s_err[current_level, current_level])
            err_min = min(err_min, s_err[current_level, current_level])
            err_mean += s_err[current_level, current_level]
            n += 1
        return (err_min, err_mean / n, err_max)


class ErrorEstimation:
    levels_of_interest = []

    def __init__(self):
        # each gas may have multiple levels of interest
        type_two = True

    def procduce_csv(self):
        # columns:
        # var | event | type (l1/l2) | err | diff | eps

        for event in range(100):
            pass
            # read original akv and avk_rc

    def avaraging_kernel(self):
        raise NotImplementedError

    def noise_matrix(self):
        raise NotImplementedError

    def cross_averaging_kernel(self):
        raise NotImplementedError


class WaterVapour:
    levels_of_interest = [-16, -20]
    # for each method type one and type two
    pass


class GreenhouseGases:
    levels_of_interest = [-10, -19]
    pass


class NitridAcid:
    levels_of_interest = [-6]
    pass
