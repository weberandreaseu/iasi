import math
import os

import luigi
import numpy as np
import pandas as pd
from netCDF4 import Dataset, Group, Variable
from sklearn.model_selection import ParameterGrid

from iasi.composition import Composition
from iasi.compression import CompressDataset, SelectSingleVariable
from iasi.file import MoveVariables
from iasi.metrics import Covariance
from iasi.quadrant import Quadrant
from iasi.util import CustomTask


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
            'threshold': [0],
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
            df = df.append({
                'gas': task.gas,
                'variable': task.variable,
                'ancestor': task.ancestor,
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
            'threshold': [0],
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
                approx_values = nc[path][...]
                original_values = original[path][...]
                report = gas_error_estimation.report_for(
                    original[path], original_values, approx_values, task.ancestor != 'MoveVariables')
                report['threshold'] = task.threshold
                report['var'] = task.variable
                gas_report = gas_report.append(report)
                nc.close()
            with self.output()[gas].open('w') as file:
                gas_report.to_csv(file, index=False)
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

    def report_for(self, variable: Variable, original, approximated, rc_error) -> pd.DataFrame:
        assert original.shape == approximated.shape
        # columns:
        # var | event | type (l1/l2) | err | diff | eps
        result = {
            'event': [],
            'level_of_interest': [],
            'err': [],
            'rc_error': [],
            'type': []
        }

        switcher = {
            'avk': self.averaging_kernel,
            'n': self.noise_matrix,
            'Tatmxavk': self.cross_averaging_kernel
        }
        error_estimation_method = switcher.get(variable.name)
        if error_estimation_method is None:
            raise ValueError(
                f'No error estimation method for variable {variable.name}')

        reshaper = Quadrant.for_assembly(variable)
        for event in range(original.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            nol_event = self.nol.data[event]
            original_event = reshaper.transform(original[event], nol_event)
            approx_event = reshaper.transform(approximated[event], nol_event)
            error_estimation_method(
                event, nol_event, original_event, approx_event, rc_error, result)
            # read original akv and avk_rc
        return pd.DataFrame(result)

    def averaging_kernel(self):
        raise NotImplementedError

    def noise_matrix(self):
        raise NotImplementedError

    def cross_averaging_kernel(self):
        raise NotImplementedError


class WaterVapour(ErrorEstimation):
    levels_of_interest = [-16, -20]
    # for each method type one and type two

    def averaging_kernel(self, event, level_event, original_event, approx_event, rc_error, result) -> None:
        for type in range(1, 3):
            if type == 1:
                e_err = self.type1_error(
                    event, level_event, original_event, approx_event, rc_error)
            elif type == 2:
                e_err = self.type2_error(
                    event, level_event, original_event, approx_event, rc_error)
            else:
                continue
            for loi in self.levels_of_interest:
                level = level_event + loi
                if level < 2:
                    continue
                result['event'].append(event)
                result['level_of_interest'].append(loi)
                result['err'].append(e_err[level, level])
                result['rc_error'].append(rc_error)
                result['type'].append(type)

    def type1_error(self, event, level_event, original_event, approx_event, rc_error):
        e_cov = Covariance(level_event, self.alt[event])
        original_type1 = e_cov.type1_of(original_event)
        if rc_error:
            approx_type1 = e_cov.type1_of(approx_event)
            e_err = e_cov.smoothing_error_covariance(
                original_type1, approx_type1)
        else:
            e_err = e_cov.smoothing_error_covariance(
                original_type1, np.identity(2 * level_event))
        return e_err

    def type2_error(self, event, level_event, original_event, approx_event, rc_error):
        e_cov = Covariance(level_event, self.alt[event])
        original_type2 = e_cov.type2_of(original_event)
        if rc_error:
            approx_type2 = e_cov.type2_of(approx_event)
            return e_cov.smoothing_error_covariance(original_type2, approx_type2)
        else:
            return e_cov.smoothing_error_covariance(original_type2, np.identity(2 * level_event))

    def noise_matrix(self, event, level_event, original_event, approx_event, rc_error, result):
        cov_event = Covariance(level_event, self.alt[event])
        # original/approx event is already covariance matrix -> only type1/2 transformation
        err_original = cov_event.type1_of(original_event)
        if rc_error:
            err = err_original - cov_event.type1_of(approx_event)
        else:
            err = err_original
        for loi in self.levels_of_interest:
            level = level_event + loi
            if level < 2:
                continue
            result['event'].append(event)
            result['level_of_interest'].append(loi)
            result['err'].append(err[level, level])
            result['rc_error'].append(rc_error)
            result['type'].append(1)


class GreenhouseGas(ErrorEstimation):
    levels_of_interest = [-10, -19]


class NitridAcid(ErrorEstimation):
    levels_of_interest = [-6]
