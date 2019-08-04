from functools import partial
import math
import os

import luigi
import numpy as np
import pandas as pd
from netCDF4 import Dataset, Group, Variable
from sklearn.model_selection import ParameterGrid

from iasi.composition import Composition
from iasi.compression import CompressDataset, SelectSingleVariable, DecompressDataset
from iasi.file import MoveVariables, FileTask
from iasi.metrics import Covariance
from iasi.quadrant import Quadrant, AssembleFourQuadrants
from iasi.util import CustomTask
import logging

logger = logging.getLogger(__name__)


class EvaluationTask(FileTask):
    gases = luigi.ListParameter()
    variables = luigi.ListParameter()
    threshold_values = luigi.ListParameter(default=[1e-2, 1e-3, 1e-4, 1e-5])
    ancestor = None

    def requires(self):
        compression_parameter = {
            'ancestor': [self.ancestor],
            'file': [self.file],
            'dst': [self.dst],
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
            'original': MoveVariables(dst=self.dst, file=self.file)
        }


class EvaluationCompressionSize(EvaluationTask):
    ancestor = 'CompressDataset'

    def output_directory(self):
        return 'compression-summary'

    def output_extension(self):
        return '.csv'

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


class EvaluationErrorEstimation(FileTask):
    file = luigi.Parameter()
    gases = luigi.Parameter()
    variables = luigi.Parameter()
    thresholds = luigi.ListParameter(default=[1e-3])

    def output_directory(self):
        return 'error-estimation'

    def output_extension(self):
        return '.csv'

    def requires(self):
        parameter = {
            'file': [self.file],
            'dst': [self.dst],
            'thresholds': [self.thresholds],
            'gas': self.gases,
            'variable': self.variables,
            'log_file': [self.log_file]
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


class VariableErrorEstimation(FileTask):

    gas = luigi.Parameter()
    variable = luigi.Parameter()
    thresholds = luigi.ListParameter(default=[1e-3])

    def output_extension(self):
        return '.csv'

    def requires(self):
        compressed = [DecompressDataset(
            dst=self.dst,
            file=self.file,
            threshold=threshold,
            log_file=self.log_file,
            compress_upstream=True
        ) for threshold in self.thresholds]
        original = MoveVariables(
            dst=self.dst, file=self.file, log_file=self.log_file)
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
        alt_trop = original['tropopause_altitude'][...]
        counter = 0
        message = f'Calculate original error for {path}: {counter}/{len(tasks_and_input)}'
        logger.info(message)
        self.set_status_message(message)
        self.set_progress_percentage(int(counter / len(tasks_and_input) * 100))
        error_estimation: ErrorEstimation = ErrorEstimation.factory(
            self.gas, nol, alt, avk, alt_trop=alt_trop)
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

    def output_directory(self):
        return os.path.join('error-estimation', self.gas, self.variable)


class ErrorEstimation:
    levels_of_interest = []
    # assume statosphere starting at 25 km
    alt_strat = 25000

    @staticmethod
    def factory(gas: str, nol, alt, avk, alt_trop=None):
        if gas == 'WV':
            return WaterVapour(gas, nol, alt, avk, alt_trop, type_two=True)
        if gas == 'GHG':
            return GreenhouseGas(gas, nol, alt, alt_trop)
        if gas == 'HNO3':
            return NitridAcid(gas, nol, alt, alt_trop)
        if gas == 'Tatm':
            return AtmosphericTemperature(gas, nol, alt, alt_trop)
        raise ValueError(f'No error estimation implementation for gas {gas}')

    def __init__(self, gas, nol, alt, alt_trop, type_two=False):
        # each gas may have multiple levels of interest
        self.type_two = type_two
        self.nol = nol
        self.alt = alt
        self.gas = gas
        self.alt_trop = alt_trop

    def matrix_ok(self, event, path, matrix):
        ok = True
        if np.ma.is_masked(matrix):
            logger.warning(
                'event %d contains masked values in %s. skipping...', event, path)
            ok = False
        if np.isnan(matrix).any():
            logger.warning(
                'event %d contains nan values in %s. skipping...', event, path)
            ok = False
        if np.isinf(matrix).any():
            logger.warning(
                'event %d contains inf values in %s. skipping...', event, path)
            ok = False
        if np.allclose(matrix, 0, atol=1e-14):
            logger.warning(
                'event %d contains zero or close to zero values in %s. skipping...', event, path)
            ok = False
        return ok

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
        path = f'/state/{self.gas}/{variable.name}'
        for event in range(original.shape[0]):
            if np.ma.is_masked(self.nol[event]) or self.nol.data[event] > 29:
                continue
            nol_event = self.nol.data[event]
            if not self.matrix_ok(event, path, self.alt[event, :nol_event]):
                continue
            covariance = Covariance(nol_event, self.alt[event])
            original_event = reshaper.transform(original[event], nol_event)
            if not self.matrix_ok(event, path, original_event):
                continue
            # use reconstruced values iff rc_error flag is set
            if rc_error:
                rc_event = reshaper.transform(reconstructed[event], nol_event)
                if not self.matrix_ok(event, path, rc_event):
                    continue
                rc_event = rc_event.data
            else:
                rc_event = None
            if isinstance(self, WaterVapour):
                avk_event = AssembleFourQuadrants(
                    nol_event).transform(self.avk[event], nol_event)
                if not self.matrix_ok(event, 'wv_avk', avk_event):
                    continue
                avk_event = avk_event.data
            else:
                avk_event = None
            # type two error only exists for water vapour
            # if gas does not require type 2 error estimation, break loop after first iteration
            calc_type_two = self.type_two
            while True:
                error = estimation_method(event,
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
                        result['err'].append(
                            error[level + nol_event, level + nol_event])
                        result['rc_error'].append(rc_error)
                        result['type'].append(2 if calc_type_two else 1)
                # stop if type 1 is calculated
                if not calc_type_two:
                    break
                # just finished type 2 in first iteration -> repeat with type 1
                calc_type_two = False
        return pd.DataFrame(result)

    def averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError

    def noise_matrix(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError

    def cross_averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        raise NotImplementedError

    def smoothing_error(self, actual_matrix, to_compare, assumed_covariance) -> np.ndarray:
        """Calulate smooting error with two matrices and assumed covariance"""
        return (actual_matrix - to_compare) @ assumed_covariance @ (actual_matrix - to_compare).T

    def assumed_covariance_temperature(self, event: int) -> np.ndarray:
        """Return assumed covariance for temperature cross averaging kernel"""
        sig = self.sigma(event)
        amp = self.amplitude_temperature(event)
        return self.construct_covariance_matrix(event, amp, sig)

    def construct_covariance_matrix(self, event, amp: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """create a covariance matrix by amplitude and deviation

        :param amp: Amplitude for levels
        :param sig: Standard deviation for levels
        """
        nol = self.nol.data[event]
        alt = self.alt.data[event]
        sa = np.ndarray((nol, nol))
        for i in range(nol):
            for j in range(nol):
                sa[i, j] = amp[i] * amp[j] * \
                    np.exp(-((alt[i] - alt[j])*(alt[i] - alt[j])) /
                           (2 * sig[i] * sig[j]))
        return sa

    def sigma(self, event, f_sigma: float = 0.6) -> np.ndarray:
        """Assumed correlation length for all gases and temperature.

        :param self.alt_strat:   altitude of stratosphere in meters
        :param f_sigma:     scaling factor

        :return: correlation length for each level
        """
        nol = self.nol.data[event]
        alt = self.alt.data[event]
        alt_trop = self.alt_trop[event]
        sig = np.ndarray(nol)
        for i in range(nol):
            # below tropopause
            if alt[i] < alt_trop:
                sig[i] = 2500 + (alt[i] - alt[0]) * \
                    ((5000-2500)/(alt_trop-alt[0]))
            # inside statrophere
            if alt[i] >= alt_trop and alt[i] < self.alt_strat:
                sig[i] = 5000+(alt[i]-alt_trop) * \
                    ((10000-5000)/(self.alt_strat-alt_trop))
            # above stratosphere
            if alt[i] > self.alt_strat:
                sig[i] = 10000
        return sig * f_sigma

    def amplitude(self, event):
        raise NotImplementedError

    def amplitude_temperature(self, event) -> np.ndarray:
        """Get amplitude and deviation for atmospheric temperature

        :return: amp
        """
        nol = self.nol.data[event]
        alt = self.alt.data[event, :nol]
        alt_trop = self.alt_trop.data[event]
        amp = np.ndarray(nol)
        for i in range(nol):
            if alt[0]+4000 < alt_trop:
                # setting amp_T
                if alt[i] <= alt[0]+4000:
                    amp[i] = 2.0 - 1.0 * (alt[i] - alt[0]) / 4000
                elif alt[i] >= alt[0]+4000 and alt[i] <= alt_trop:
                    amp[i] = 1.
                elif alt[i] > alt_trop and alt[i] <= alt_trop+5000:
                    amp[i] = 1.0 + 0.5 * (alt[i] - alt_trop) / 5000
                elif alt[i] > alt_trop+5000:
                    amp[i] = 1.5
            else:
                # setting amp[i]
                if alt[i] < alt_trop:
                    amp[i] = 2.0 - 1.0 * (alt[i] - alt[0]) / \
                        (alt_trop - alt[0])
                elif alt[i] == alt_trop:
                    amp[i] = 1.
                elif alt[i] > alt_trop and alt[i] <= alt_trop+5000:
                    amp[i] = 1.0 + 0.5 * (alt[i] - alt_trop) / 5000
                elif alt[i] > alt_trop+5000:
                    amp[i] = 1.5
        return amp


class WaterVapour(ErrorEstimation):
    levels_of_interest = [-6, -16, -19]

    def __init__(self, gas, nol, alt, avk, alt_trop, type_two=True):
        super().__init__(gas, nol, alt, alt_trop, type_two=type_two)
        self.avk = avk

    # for each method type one and type two

    def averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        # in this method, avk should be same like original
        if not np.allclose(original, avk):
            logger.warn('There are differences in original parameter and avk')
        s_cov = self.assumed_covariance(event)
        nol = self.nol.data[event]
        if type2:
            # type 2 error
            original_type2 = covariance.type2_of(original)
            if reconstructed is None:
                # type 2 original error
                return self.smoothing_error(original_type2, np.identity(2 * nol), s_cov)
            else:
                # type 2 reconstruction error
                rc_type2 = covariance.type2_of(reconstructed)
                return self.smoothing_error(original_type2, rc_type2, s_cov)
        else:
            # type 1 error
            original_type1 = covariance.type1_of(original)
            if reconstructed is None:
                # type 1 original error
                return self.smoothing_error(
                    original_type1, np.identity(2 * nol), s_cov)
            else:
                # type 1 reconstruction error
                rc_type1 = covariance.type1_of(reconstructed)
                return self.smoothing_error(original_type1, rc_type1, s_cov)

    def noise_matrix(self, event: int, original: np.ndarray, reconstruced: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        # original/approx event is already covariance matrix -> only type1/2 transformation
        assert avk is not None
        P = covariance.traf()
        if type2:
            # type 2 error
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

    def cross_averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert avk is not None
        P = covariance.traf()
        s_cov = self.assumed_covariance_temperature(event)
        if type2:
            # type 2 error
            C = covariance.c_by_avk(avk)
            original_type2 = C @ P @ original
            if reconstructed is None:
                # original error
                return original_type2 @ s_cov @ original_type2.T
            # reconstruction error
            rc_type2 = C @ P @ reconstructed
            return self.smoothing_error(original_type2, rc_type2, s_cov)
        else:
            # type 1 error
            original_type1 = P @ original
            if reconstructed is None:
                # original error
                return original_type1 @ s_cov @ original_type1.T
            else:
                # reconstruction error
                rc_type1 = P @ reconstructed
                return self.smoothing_error(original_type1, rc_type1, s_cov)

    def assumed_covariance(self, event: int) -> np.ndarray:
        """Assumed covariance for both H2O and HDO"""
        nol = self.nol.data[event]
        amp_H2O, amp_dD = self.amplitude(event)
        sig = self.sigma(event)
        Sa_ = np.zeros([2*nol, 2*nol])
        # Sa H2O
        Sa_[:nol, :nol] = self.construct_covariance_matrix(event, amp_H2O, sig)
        # Sa delD
        Sa_[nol:, nol:] = self.construct_covariance_matrix(event, amp_dD, sig)
        return Sa_

    def amplitude(self, event):
        """Calculate amplitude for H2O and HDO

        :return: (amp_H2O, amp_dD)
        """
        nol = self.nol.data[event]
        alt = self.alt.data[event, :nol]
        alt_trop = self.alt_trop.data[event]
        amp_H2O = np.ndarray(nol)
        amp_dD = np.ndarray(nol)
        for i in range(nol):
            if alt[i] < 5000.:
                amp_H2O[i] = 0.75 * (1 + alt[i] / 5000)
                amp_dD[i] = 0.09 * (1 + alt[i] / 5000)
            elif 5000. <= alt[i] < alt_trop:
                amp_H2O[i] = 1.5
                amp_dD[i] = 0.18
            elif alt_trop <= alt[i] < self.alt_strat:
                amp_H2O[i] = 1.5 - 1.2 * \
                    (alt[i] - alt_trop) / (self.alt_strat - alt_trop)
                amp_dD[i] = 0.18 - 0.12 * \
                    (alt[i] - alt_trop) / (self.alt_strat - alt_trop)
            elif alt[i] >= self.alt_strat:
                amp_H2O[i] = 0.3
                amp_dD[i] = 0.06
            else:
                raise ValueError(f'Invalid altitude at {event}')
        return amp_H2O, amp_dD


class GreenhouseGas(ErrorEstimation):
    levels_of_interest = [-6, -10, -19]

    def averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # original error
            reconstructed = np.identity(covariance.nol * 2)
        s_cov = self.assumed_covariance(event)
        return self.smoothing_error(original, reconstructed, s_cov)

    def cross_averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        s_cov = self.assumed_covariance_temperature(event)
        if reconstructed is None:
            # original error
            return original @ s_cov @ original.T
        return self.smoothing_error(original, reconstructed, s_cov)

    def noise_matrix(self,  event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)

    def assumed_covariance(self, event) -> np.ndarray:
        amp = self.amplitude(event)
        sig = self.sigma(event)
        s_cov = self.construct_covariance_matrix(event, amp, sig)
        nol = self.nol.data[event]
        s_cov_ghg = np.zeros((2 * nol, 2 * nol))
        s_cov_ghg[:nol, :nol] = s_cov
        s_cov_ghg[nol:, nol:] = s_cov
        return s_cov_ghg

    def amplitude(self, event) -> np.ndarray:
        """Amplitude for GHG"""
        nol = self.nol.data[event]
        alt = self.alt.data[event, :nol]
        alt_trop = self.alt_trop.data[event]
        amp = np.ndarray((nol))
        for i in range(nol):
            if alt[i] < alt_trop:
                amp[i] = 0.1
            elif alt_trop <= alt[i] < self.alt_strat:
                amp[i] = 0.1 + (alt[i] - alt_trop) * \
                    ((0.25 - 0.1)/(self.alt_strat - alt_trop))
            elif alt[i] >= self.alt_strat:
                amp[i] = 0.25
            else:
                raise ValueError('Invalid altitude')
        return amp


class NitridAcid(ErrorEstimation):
    levels_of_interest = [-6]

    def averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        assert not type2
        if reconstructed is None:
            # original error
            reconstructed = np.identity(covariance.nol)
        s_cov = self.assumed_covariance(event)
        return self.smoothing_error(original, reconstructed, s_cov)

    def cross_averaging_kernel(self,  event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        s_cov = self.assumed_covariance_temperature(event)
        if reconstructed is None:
            # original error
            return original @ s_cov @ original.T
        return self.smoothing_error(original, reconstructed, s_cov)

    def noise_matrix(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None) -> np.ndarray:
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)

    def assumed_covariance(self, event) -> np.ndarray:
        amp = self.amplitude(event)
        sig = self.sigma(event)
        return self.construct_covariance_matrix(event, amp, sig)

    def amplitude(self, event: int):
        """Amplitude of HNO3"""
        nol = self.nol.data[event]
        alt = self.alt.data[event, :nol]
        alt_trop = self.alt_trop.data[event]
        amp = np.ndarray((nol))
        for i in range(nol):
            # surface is more than 4km below tropopause
            if alt[0] < alt_trop - 4000:
                # higher variances in valley's due to human made emmisions
                if alt[i] < alt_trop - 4000:
                    amp[i] = 2.4 + (alt[i] - alt[0]) * \
                        ((1.2 - 2.4)/(alt_trop - 4000 - alt[0]))
                elif alt_trop - 4000 <= alt[i] < alt_trop + 8000:
                    amp[i] = 1.2
                elif alt_trop + 8000 <= alt[i] < 50000:
                    amp[i] = 1.2 + (alt[i] - (alt_trop + 8000)) * \
                        ((0.3-1.2) / (50000 - (alt_trop + 8000)))
                elif alt[i] >= 50000:
                    amp[i] = 0.3
                else:
                    raise ValueError('Invalid altitude')
            else:
                # at higher altitudes covariance is lower
                if alt_trop - 4000 <= alt[i] < alt_trop + 8000:
                    amp[i] = 1.2
                elif alt_trop + 8000 < alt[i] < 50000:
                    amp[i] = 1.2 + (alt[i] - (alt_trop + 8000)) * \
                        ((0.3 - 1.2)/(50000 - (alt_trop + 8000)))
                elif alt[i] >= 50000:
                    amp[i] = 0.3
                else:
                    raise ValueError('Invalid altitude')
        return amp


class AtmosphericTemperature(ErrorEstimation):
    # zero means surface
    levels_of_interest = [0, -10, -19]

    def averaging_kernel(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        assert not type2
        if reconstructed is None:
            reconstructed = np.identity(covariance.nol)
        s_cov = self.assumed_covariance_temperature(event)
        return self.smoothing_error(original, reconstructed, s_cov)

    def noise_matrix(self, event: int, original: np.ndarray, reconstructed: np.ndarray, covariance: Covariance, type2=False, avk=None):
        assert not type2
        if reconstructed is None:
            return original
        else:
            return np.absolute(original - reconstructed)
