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
import logging

logger = logging.getLogger(__name__)


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
        single_variables = tasks + \
            [SelectSingleVariable(**params)
             for params in uncompressed_param_grid]
        # exclude cross average kernel from atmospheric temperature.
        # atmospheric temperature has only avk and noise matrix
        filtered = filter(lambda task: not(task.gas == 'Tatm') or not(
            task.variable == 'Tatmxavk'), single_variables)
        return {
            'single': filtered,
            'original': MoveVariables(dst=self.dst, file=self.file, force=self.force, force_upstream=self.force_upstream)
        }

    def run(self):
        tasks_and_input = list(zip(
            self.requires()['single'], self.input()['single']))
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
        if gas == 'Tatm':
            return AtmosphericTemperature(nol, alt)
        raise ValueError(f'No error estimation implementation for gas {gas}')

    def __init__(self, nol, alt, type_two=False):
        # each gas may have multiple levels of interest
        self.type_two = type_two
        self.nol = nol
        self.alt = alt

    def report_for(self, variable: Variable, original, reconstructed, rc_error) -> pd.DataFrame:
        assert original.shape == reconstructed.shape
        result = {
            'event': [],
            'level_of_interest': [],
            'err': [],
            'rc_error': [],
            'type': []
        }
        error_estimation_methods = {
            'avk': self.averaging_kernel,
            'n': self.noise_matrix,
            'Tatmxavk': self.cross_averaging_kernel
        }
        estimation_method = error_estimation_methods.get(variable.name)
        if estimation_method is None:
            raise ValueError(
                f'No error estimation method for variable {variable.name}')

        reshaper = Quadrant.for_assembly(variable)
        for event in range(original.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            nol_event = self.nol.data[event]
            covariance = Covariance(nol_event, self.alt[event])
            original_event = reshaper.transform(original[event], nol_event)
            if original_event.mask.any():
                logger.warn('Original array contains masked values')
            # use reconstruced values iff rc_error flag is set
            if rc_error:
                rc_event = reshaper.transform(reconstructed[event], nol_event)
                if rc_event.mask.any():
                    logger.warn('Reconstructed array contains masked values')
                rc_event = rc_event.data
            else:
                rc_event = None

            # type two error only exists for water vapour
            # if gas does not require type 2 error estimation, break loop after first iteration
            calc_type_two = self.type_two
            while True:
                error = estimation_method(
                    original_event.data, rc_event, covariance, type2=calc_type_two)
                self.add_error_to_report(
                    result, nol_event, event, error, rc_error, type_two=calc_type_two)
                # stop if type 1 is calculated
                if not calc_type_two:
                    break
                # just finished type 2 in first iteration -> repeat with type 1
                calc_type_two = False
        return pd.DataFrame(result)

    def add_error_to_report(self, result, nol, event, error, rc_error, type_two=False):
        for loi in self.levels_of_interest:
            level = nol + loi
            if level < 2:
                continue
            result['event'].append(event)
            result['level_of_interest'].append(loi)
            result['err'].append(error[level, level])
            result['rc_error'].append(rc_error)
            if type_two:
                result['type'].append(2)
            else:
                result['type'].append(1)

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False):
        raise NotImplementedError

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False):
        raise NotImplementedError

    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False):
        raise NotImplementedError


class WaterVapour(ErrorEstimation):
    levels_of_interest = [-16, -20]
    # for each method type one and type two

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        rc_error = reconstructed is not None
        if type2:
            # type 2 error
            original_type2 = covariance.type2_of(original)
            if reconstructed is None:
                # type 2 original error
                return covariance.smoothing_error(original_type2, np.identity(2 * covariance.nol))
            else:
                # type 2 reconstruction error
                rc_type2 = covariance.type2_of(reconstructed)
                return covariance.smoothing_error(original_type2, rc_type2)
        else:
            # type 1 error
            original_type1 = covariance.type1_of(original)
            if reconstructed is None:
                # type 1 original error
                return covariance.smoothing_error(
                    original_type1, np.identity(2 * covariance.nol))
            else:
                # type 1 reconstruction error
                rc_type1 = covariance.type1_of(reconstructed)
                return covariance.smoothing_error(original_type1, rc_type1)

    def noise_matrix(self, original_event, approx_event, covariance, type2=False) -> np.ndarray:
        # original/approx event is already covariance matrix -> only type1/2 transformation
        err_original = covariance.type1_of(original_event)
        if approx_event is None:
            return err_original
        else:
            return err_original - covariance.type1_of(approx_event)

    def cross_averaging_kernel(self, original_event, approx_event, covariance, type2=False) -> np.ndarray:
        # original_type1 = covariance.type1_of(original_event)
        P = covariance.traf()
        original_type1 = P @ original_event.data
        s_cov = covariance.type1_covariance()[:covariance.nol, :covariance.nol]

        if approx_event is None:
            # TODO: what is the ideal cross averaging kernel?
            # original error
            to_compare = np.identity(covariance.nol * 2)[:, :covariance.nol]
        else:
            # reconstruction error
            to_compare = P @ approx_event.data
        return (original_type1 - to_compare) @ s_cov @ (original_type1 - to_compare).T


class GreenhouseGas(ErrorEstimation):
    levels_of_interest = [-10, -19]

    # TODO validate
    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        assert not type2
        original_type1 = covariance.type1_of(original)
        if reconstructed is None:
            # original error
            identity = np.identity(covariance.nol * 2)
            return covariance.smoothing_error(original_type1, identity)
        else:
            # reconstruction error
            rc_type1 = covariance.type1_of(reconstructed)
            return covariance.smoothing_error(original_type1, rc_type1)

    # TODO validate
    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        P = covariance.traf()
        original_type1 = P @ original
        s_cov = covariance.type1_covariance()[:covariance.nol, :covariance.nol]

        if reconstructed is None:
            # TODO: what is the ideal cross averaging kernel?
            # original error
            to_compare = np.identity(covariance.nol * 2)[:, :covariance.nol]
        else:
            # reconstruction error
            to_compare = P @ reconstructed.data
        return (original_type1 - to_compare) @ s_cov @ (original_type1 - to_compare).T

    # TODO validate
    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        original_type1 = covariance.type1_of(original)
        if reconstructed is None:
            return original_type1
        else:
            return original_type1 - covariance.type1_of(reconstructed)


class NitridAcid(ErrorEstimation):
    levels_of_interest = [-6]

    # TODO validate
    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # original error
            to_compare = np.identity(covariance.nol)
        else:
            # reconstruction error
            to_compare = reconstructed
        s_cov = covariance.type1_covariance()[:covariance.nol, :covariance.nol]
        return (original - to_compare) @ s_cov @ (original - to_compare).T

    # TODO validate
    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        s_cov = covariance.type1_covariance()[:covariance.nol, :covariance.nol]
        if reconstructed is None:
            # TODO: what is the ideal cross averaging kernel?
            # original error
            to_compare = np.identity(covariance.nol)
        else:
            # reconstruction error
            to_compare = reconstructed
        return (original - to_compare) @ s_cov @ (original - to_compare).T

    # TODO validate
    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False) -> np.ndarray:
        if reconstructed is None:
            return original
        else:
            return original - reconstructed


class AtmosphericTemperature(ErrorEstimation):
    # TODO what are levels of interest?
    levels_of_interest = [-10]

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False):
        assert not type2
        if reconstructed is None:
            return original - np.identity(covariance.nol)
        else:
            return original - reconstructed

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False):
        assert not type2
        if reconstructed is None:
            return original
        else:
            return original - reconstructed
