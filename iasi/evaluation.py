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
from iasi.quadrant import Quadrant


class EvaluationTask(luigi.Config, CustomTask):
    file = luigi.Parameter()
    gases = luigi.ListParameter()
    variables = luigi.ListParameter()


class EvaluationCompressionSize(EvaluationTask):

    def requires(self):
        compression_parameter = {
            'ancestor': ['CompressDataset'],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'force_upstream': [self.force_upstream],
            'threshold': [1e-2, 1e-3, 1e-4, 1e-5],
            'gas': self.gases,
            'variable': self.variables
        }
        compressed_param_grid = list(ParameterGrid(compression_parameter))
        tasks = [SelectSingleVariable(**params)
                 for params in compressed_param_grid]
        # for uncompressed dataset we do not need multiple threshold values
        uncompressed_parameter = {
            'ancestor': ['MoveVariables'],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'threshold': [np.nan],
            'gas': self.gases,
            'variable': self.variables
        }
        uncompressed_param_grid = list(ParameterGrid(uncompressed_parameter))
        return {
            'single': tasks + [SelectSingleVariable(**params) for params in uncompressed_param_grid],
            'original': MoveVariables(dst=self.dst, file=self.file)
        }

    def output(self):
        return self.create_local_target('compression-summary', file=self.file, ext='csv')

    def size_in_kb(self, file):
        return int(os.path.getsize(file) / (1000))

    def run(self):
        # get size for all parameters
        df = pd.DataFrame()
        for task, input in zip(self.requires()['single'], self.input()['single']):
            print(input.path)
            print(task.param_kwargs)
            df = df.append({
                'variable': task.variable,
                'compressed': task.ancestor,
                'size': self.size_in_kb(input.path),
                'threshold': task.threshold
            }, ignore_index=True)
        df.to_csv(self.output().path, index=False)


class EvaluationErrorEstimation(EvaluationTask):
    def requires(self):
        compression_parameter = {
            'ancestor': ['DecompressDataset'],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'force_upstream': [self.force_upstream],
            'threshold': [1e-2, 1e-3, 1e-4, 1e-5],
            'gas': self.gases,
            'variable': self.variables
        }
        compressed_param_grid = list(ParameterGrid(compression_parameter))
        tasks = [SelectSingleVariable(**params)
                 for params in compressed_param_grid]
        # for uncompressed dataset we do not need multiple threshold values
        uncompressed_parameter = {
            'ancestor': ['MoveVariables'],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'force_upstream': [self.force_upstream],
            'threshold': [np.nan],
            'gas': self.gases,
            'variable': self.variables
        }
        uncompressed_param_grid = list(ParameterGrid(uncompressed_parameter))
        return {
            'single': tasks + [SelectSingleVariable(**params) for params in uncompressed_param_grid],
            'original': MoveVariables(dst=self.dst, file=self.file, force=self.force, force_upstream=self.force_upstream)
        }

    def run(self):
        tasks_and_input = zip(
            self.requires()['single'], self.input()['single'])
        original = Dataset(self.input()['original'].path)
        nol = original['atm_nol'][...]
        alt = original['atm_altitude'][...]
        for gas in self.gases:
            gas_report = pd.DataFrame()
            # group tasks and input by gas
            gas_task_and_input = filter(
                lambda tai: tai[0].gas == gas, tasks_and_input)
            gas_error_estimation = ErrorEstimation.factory(gas, nol, alt)
            for task, input in gas_task_and_input:
                # output_df, task, gas, variables
                nc = Dataset(input.path)
                var = task.variable
                path = f'/state/{gas}/{var}'
                if isinstance(nc[path], Group):
                    # variable is decomposed
                    approx_values = Composition.factory(
                        nc[path]).reconstruct(nol)
                else:
                    approx_values = nc[path][...]
                original_values = original[path][...]
                report = gas_error_estimation.report_for(
                    var, original_values, approx_values)
                # TODO extend report
                nc.close()
            with self.output()[gas].open('w') as file:
                gas_report.to_csv(file)
        original.close()

    def output(self):
        # one error estimation report for each gas
        return {gas: self.create_local_target('error-estimation', gas, file=self.file, ext='csv') for gas in self.gases}


class ErrorEstimation:
    levels_of_interest = []

    @staticmethod
    def factory(gas: str, nol, alt):
        if gas == 'WV':
            return WaterVapour(nol, alt, type_two=True)
        if gas == 'GHG':
            return GreenhouseGas(nol, alt)
        if gas == 'HNO3':
            return NitridAcid(nol, alt)
        raise ValueError(f'No error estimation implementation for gas {gas}')

    def __init__(self, nol, alt, type_two=False):
        # each gas may have multiple levels of interest
        self.type_two = type_two
        self.nol = nol
        self.alt = alt

    def report_for(self, variable, original, approximated) -> pd.DataFrame:
        assert original.shape == approximated.shape
        # columns:
        # var | event | type (l1/l2) | err | diff | eps
        df = pd.DataFrame()
        for event in range(original.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            e_nol = self.nol.data[event]
            e_cov = Covariance(e_nol, self.alt[event])
            # read original akv and avk_rc

    def avaraging_kernel(self):
        raise NotImplementedError

    def noise_matrix(self):
        raise NotImplementedError

    def cross_averaging_kernel(self):
        raise NotImplementedError


class WaterVapour(ErrorEstimation):
    levels_of_interest = [-16, -20]
    # for each method type one and type two


class GreenhouseGas(ErrorEstimation):
    levels_of_interest = [-10, -19]


class NitridAcid(ErrorEstimation):
    levels_of_interest = [-6]

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
