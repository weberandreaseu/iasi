import math
import os

import luigi
import numpy as np
import pandas as pd
from netCDF4 import Dataset, Group, Variable
from sklearn.model_selection import ParameterGrid

from iasi.composition import Composition
from iasi.compression import CompressDataset, SelectSingleVariable, DecompressDataset
from iasi.file import MoveVariables
from iasi.metrics import Covariance
from iasi.quadrant import Quadrant, AssembleFourQuadrants
from iasi.util import CustomTask
import logging

logger = logging.getLogger(__name__)


class EvaluationTask(CustomTask):
    file = luigi.Parameter()
    gases = luigi.ListParameter()
    variables = luigi.ListParameter()
    threshold_values = luigi.ListParameter(default=[1e-2, 1e-3, 1e-4, 1e-5])
    ancestor = None

    def requires(self):
        compression_parameter = {
            'ancestor': [self.ancestor],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'force_upstream': [self.force_upstream],
            'threshold': self.threshold_values,
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


class EvaluationCompressionSize(EvaluationTask):
    ancestor = 'CompressDataset'

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
        with self.output().temporary_path() as target:
            df.to_csv(target, index=False)


class EvaluationErrorEstimation(CustomTask):
    file = luigi.Parameter()
    gases = luigi.Parameter()
    variables = luigi.Parameter()
    thresholds = luigi.ListParameter(default=[1e-3])

    def requires(self):
        parameter = {
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'force_upstream': [self.force_upstream],
            'thresholds': [self.thresholds],
            'gas': self.gases,
            'variable': self.variables,
            'log': [self.log]
        }
        parameter_grid = ParameterGrid(parameter)
        # exclude cross average kernel from atmospheric temperature.
        # atmospheric temperature has only avk and noise matrix
        parameter_grid = filter(lambda params: not(params['gas'] == 'Tatm') or not(
            params['variable'] == 'Tatmxavk'), parameter_grid)
        return [VariableErrorEstimation(**params) for params in parameter_grid]

    def run(self):
        report = pd.DataFrame()
        for task in self.input():
            with task.open() as file:
                task_report = pd.read_csv(file)
                report = report.append(task_report)
        with self.output().temporary_path() as target:
            report.to_csv(target, index=False)

    def output(self):
        return self.create_local_target('error-estimation', file=self.file, ext='csv')


class VariableErrorEstimation(CustomTask):

    file = luigi.Parameter()
    gas = luigi.Parameter()
    variable = luigi.Parameter()
    thresholds = luigi.ListParameter(default=[1e-3])

    def requires(self):
        compressed = [DecompressDataset(
            dst=self.dst,
            file=self.file,
            threshold=threshold,
            force=self.force,
            force_upstream=self.force_upstream,
            log=self.log
        ) for threshold in self.thresholds]
        original = MoveVariables(dst=self.dst,
                                 file=self.file,
                                 log=self.log,
                                 force=self.force,
                                 force_upstream=self.force_upstream)
        return {
            'compressed': compressed,
            'original': original
        }

    def run(self):
        path = f'/state/{self.gas}/{self.variable}'
        logger.info('Starting error estimation for %s', path)
        tasks_and_input = list(zip(
            self.requires()['compressed'], self.input()['compressed']))
        original = Dataset(self.input()['original'].path)
        nol = original['atm_nol'][...]
        alt = original['atm_altitude'][...]
        avk = original['/state/WV/avk'][...]
        counter = 0
        message = f'Calculate original error for {path}: {counter}/{len(tasks_and_input)}'
        logger.info(message)
        self.set_status_message(message)
        self.set_progress_percentage(int(counter / len(tasks_and_input) * 100))
        error_estimation: ErrorEstimation = ErrorEstimation.factory(
            self.gas, nol, alt, avk)
        # calculation of original error
        variable_report = error_estimation.report_for(
            original[path], original[path][...], None, rc_error=False)
        variable_report['threshold'] = 0
        # calculation of reconstruction error
        for task, input in tasks_and_input:
            counter += 1
            nc = Dataset(input.path)
            message = f'Calculating error estimation {counter} of {len(tasks_and_input)} for {path} with threshold {task.threshold}'
            logger.info(message)
            self.set_status_message(message)
            self.set_progress_percentage(
                int(counter / len(tasks_and_input) * 100))
            reconstructed_values = nc[path][...]
            original_values = original[path][...]
            report = error_estimation.report_for(
                original[path], original_values, reconstructed_values, rc_error=True)
            report['threshold'] = task.threshold
            variable_report = variable_report.append(report, ignore_index=True)
            nc.close()
        variable_report['var'] = self.variable
        variable_report['gas'] = self.gas
        with self.output().temporary_path() as target:
            variable_report.to_csv(target, index=False)
        original.close()

    def output(self):
        # one error estimation report for each gas
        return self.create_local_target('error-estimation', self.gas, self.variable, file=self.file, ext='csv')

    def log_file(self):
        return self.create_local_path('error-estimation', self.gas, self.variable, file=self.file, ext='log')


class ErrorEstimation:
    levels_of_interest = []

    @staticmethod
    def factory(gas: str, nol, alt, avk):
        if gas == 'WV':
            return WaterVapour(gas, nol, alt, avk, type_two=True)
        if gas == 'GHG':
            return GreenhouseGas(gas, nol, alt)
        if gas == 'HNO3':
            return NitridAcid(gas, nol, alt)
        if gas == 'Tatm':
            return AtmosphericTemperature(gas, nol, alt)
        raise ValueError(f'No error estimation implementation for gas {gas}')

    def __init__(self, gas, nol, alt, type_two=False):
        # each gas may have multiple levels of interest
        self.type_two = type_two
        self.nol = nol
        self.alt = alt
        self.gas = gas

    def report_for(self, variable: Variable, original, reconstructed, rc_error) -> pd.DataFrame:
        # if not original.shape == reconstructed.shape:
        #     message = f'Different shape for {type(self).__name__} {variable.name}: original {original.shape}, reconstructed {reconstructed.shape}'
        #     logger.error(message)
        #     raise ValueError(message)
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

        reshaper = Quadrant.for_assembly(self.gas, variable.name, variable)
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
            if isinstance(self, WaterVapour):
                avk_event = AssembleFourQuadrants(
                    nol_event).transform(self.avk[event], nol_event)
                if avk_event.mask.any():
                    logger.warn('Original avk contains masked values')
                avk_event = avk_event.data
            else:
                avk_event = None
            # type two error only exists for water vapour
            # if gas does not require type 2 error estimation, break loop after first iteration
            calc_type_two = self.type_two
            while True:
                error = estimation_method(
                    original_event.data, rc_event, covariance, type2=calc_type_two, avk=avk_event)
                for loi in self.levels_of_interest:
                    # zero == surface (special value)
                    if loi == 0:
                        level = 0
                    # for other levels substract from highest level
                    else:
                        level = nol_event + loi
                        if level < 2:
                            continue
                    result['event'].append(event)
                    result['level_of_interest'].append(loi)
                    result['err'].append(error[level, level])
                    result['rc_error'].append(rc_error)
                    result['type'].append(2 if calc_type_two else 1)
                    if self.gas == 'GHG':
                        # for greenhouse gases export also CH4 (lower right quadrant)
                        # nol as index offset for error level
                        result['event'].append(event)
                        result['level_of_interest'].append(loi - 29)
                        result['err'].append(error[level + nol_event, level + nol_event])
                        result['rc_error'].append(rc_error)
                        result['type'].append(2 if calc_type_two else 1)
                # stop if type 1 is calculated
                if not calc_type_two:
                    break
                # just finished type 2 in first iteration -> repeat with type 1
                calc_type_two = False
        return pd.DataFrame(result)

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError

    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError


class WaterVapour(ErrorEstimation):
    levels_of_interest = [-16, -19]

    def __init__(self, gas, nol, alt, avk, type_two=True):
        super().__init__(gas, nol, alt, type_two=type_two)
        self.avk = avk

    # for each method type one and type two
    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        # in this method, avk should be same like original
        if not np.allclose(original, avk):
            logger.warn('There are differences in original paarmeter and avk')
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

    def noise_matrix(self, original: np.ndarray, reconstruced: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        # original/approx event is already covariance matrix -> only type1/2 transformation
        assert avk is not None
        P = covariance.traf()
        if type2:
            # type 2 error
            # TODO verify correct transformation
            C = covariance.c_by_avk(avk)
            original_type2 = C @ P @ original @ P.T @ C.T
            if reconstruced is None:
                # original error
                return original_type2
            else:
                # reconstruction error
                rc_type2 = C @ P @ reconstruced @ P.T @ C.T
                return np.absolute(original_type2 - rc_type2)
        else:
            # type 1 error
            original_type1 = P @ original @ P.T
            if reconstruced is None:
                return original_type1
            else:
                rc_type1 = P @ reconstruced @ P.T
                return np.absolute(original_type1 - rc_type1)

    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert avk is not None
        P = covariance.traf()
        s_cov = covariance.assumed_covariance(species=1)
        if type2:
            # type 2 error
            C = covariance.c_by_avk(avk)
            original_type2 = C @ P @ original
            if reconstructed is None:
                # original error
                return original_type2 @ s_cov @ original_type2.T
            # reconstruction error
            rc_type2 = C @ P @ reconstructed
            return covariance.smoothing_error(original_type2, rc_type2, species=1)
        else:
            # type 1 error
            original_type1 = P @ original
            if reconstructed is None:
                # original error
                return original_type1 @ s_cov @ original_type1.T
            else:
                # reconstruction error
                rc_type1 = P @ reconstructed
                return covariance.smoothing_error(original_type1, rc_type1, species=1)


class GreenhouseGas(ErrorEstimation):
    levels_of_interest = [-6, -10, -19]

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # original error
            reconstructed = np.identity(covariance.nol * 2)
        return covariance.smoothing_error(original, reconstructed, species=2, w2=1)

    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # TODO: what is the ideal cross averaging kernel?
            # original error
            s_cov = covariance.assumed_covariance(species=1)
            return original @ s_cov @ original.T
        return covariance.smoothing_error(original, reconstructed, species=1)

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)


class NitridAcid(ErrorEstimation):
    levels_of_interest = [-6]

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # original error
            reconstructed = np.identity(covariance.nol)
        return covariance.smoothing_error(original, reconstructed, species=1)

    def cross_averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        if reconstructed is None:
            # original error
            s_cov = covariance.assumed_covariance(species=1)
            return original @ s_cov @ original.T
        return covariance.smoothing_error(original, reconstructed, species=1)

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)


class AtmosphericTemperature(ErrorEstimation):
    # zero means surface
    levels_of_interest = [0, -10, -19]

    def averaging_kernel(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        assert not type2
        if reconstructed is None:
            reconstructed = np.identity(covariance.nol)
        return covariance.smoothing_error(original, reconstructed, species=1)

    def noise_matrix(self, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        assert not type2
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)
