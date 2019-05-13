import logging
import os
import re

import luigi
from luigi import Config
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Group, Variable

import iasi
from iasi.util import CustomTask, child_groups_of, child_variables_of

logger = logging.getLogger(__name__)


def filename_by(path: str) -> str:
    _, file = os.path.split(path)
    filename, _ = os.path.splitext(file)
    return filename


class ReadFile(luigi.ExternalTask):
    """Basic class for reading a local file as input for a luigi task."""
    file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.file)


@requires(ReadFile)
class FileTask(CustomTask):
    """Base class that operates on files.

    Attributes:
        file     input file
        dst      base directory for output
        log      write task logs to file
    """
    dst = luigi.Parameter()
    log = luigi.BoolParameter(significant=False, default=False)

    def output(self):
        filename, extension = os.path.splitext(self.file)
        _, filename = os.path.split(filename)
        file = filename + (self.output_extension()
                           if self.output_extension() else extension)
        path = os.path.join(self.dst, self.output_directory(), file)
        return luigi.LocalTarget(path=path)

    def output_directory(self) -> str:
        """This directory is used by output() and logging to specify
        target of processed file.
        """
        raise NotImplementedError

    def output_extension(self) -> str:
        """Override this method if file extension of output differs from input file
        """
        return None

    @luigi.Task.event_handler(luigi.Event.START)
    def callback_start(self):
        if self.log:
            # log file destination has to be implemented by concrete task
            filename = filename_by(self.file)
            file = os.path.join(self.dst,
                                self.output_directory(),
                                filename + '.log')
            # create logging directory if non existent
            os.makedirs(os.path.join(
                self.dst, self.output_directory()), exist_ok=True)
            self.log_handler = logging.FileHandler(file, mode='w')
            self.log_handler.setFormatter(iasi.log_formatter)
            logging.getLogger().addHandler(self.log_handler)
        super(FileTask, self).callback_start()

    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def callback_execution_time(self, execution_time):
        logger.info('Task %s executeted in %s seconds', type(
            self).__name__, str(execution_time))

    @luigi.Task.event_handler(luigi.Event.SUCCESS)
    def callback_success(self):
        logger.info('Task %s successfully finished', type(self).__name__)
        self._remove_log_hander()

    @luigi.Task.event_handler(luigi.Event.FAILURE)
    def callback_failure(self, error):
        logger.error('Task %s failed', type(self).__name__)
        logger.error('Message: %s', error)
        self._remove_log_hander()

    def _remove_log_hander(self):
        if hasattr(self, 'log_handler'):
            self.log_handler.close()
            logging.getLogger().removeHandler(self.log_handler)


@requires(ReadFile)
class CopyNetcdfFile(FileTask):
    """Luigi Task for copying netCDF files with a subset of variables

    Attributes:
        format              used netCDF format
    """
    format = luigi.Parameter(default='NETCDF4')

    mapping = {
        'cld_cov': '/cld/cov',
        'cld_tp': '/cld/tp',
        'cld_ph': '/cld/ph',
        'cld_tt': '/cld/tt',
        'flg_cdlfrm': '/flg/cldfrm',
        'flg_cldnes': '/flg/cldnes',
        'flg_dustcld': '/flg/dustcld',
        'flg_iasicld': '/flg/iasicld',
        'flg_initia': '/flg/initia',
        'flg_itconv': '/flg/itconv',
        'flg_numit': '/flg/numit',
        'flg_resid': '/flg/resid',
        'flg_retcheck': '/flg/retcheck',
        'srf_flag': '/flg/srf',
        'sfc_emi': '/emi/emi',
        'sfc_emi_wn': '/emi/wn',
        'sfc_emi_flag': '/emi/flag',
        'state_GHGatm': '/state/GHG/r',
        'state_GHGatm_a': '/state/GHG/a',
        'state_GHGatm_avk': '/state/GHG/avk',
        'state_GHGatm_n': '/state/GHG/n',
        'state_Tatm2GHGatm_xavk': '/state/GHG/Tatmxavk',
        'state_Tskin2GHGatm_xavk': '/state/GHG/Tskinxavk',
        'state_HNO3atm': '/state/HNO3/r',
        'state_HNO3atm_a': '/state/HNO3/a',
        'state_HNO3atm_avk': '/state/HNO3/avk',
        'state_HNO3atm_n': '/state/HNO3/n',
        'state_Tatm2HNO3atm_xavk': '/state/HNO3/Tatmxavk',
        'state_Tskin2HNO3atm_xavk': '/state/HNO3/Tskinxavk',
        'state_Tatm': '/state/Tatm/r',
        'state_Tatm_a': '/state/Tatm/a',
        'state_Tatm_avk': '/state/Tatm/avk',
        'state_Tatm_n': '/state/Tatm/n',
        'state_Tskin2Tatm_xavk': '/state/Tatm/Tskinxavk',
        'state_Tskin': '/state/Tskin/r',
        'state_Tskin_a': '/state/Tskin/a',
        'state_Tskin_n': '/state/Tskin/n',
        'state_WVatm': '/state/WV/r',
        'state_WVatm_a': '/state/WV/a',
        'state_WVatm_avk': '/state/WV/avk',
        'state_WVatm_n': '/state/WV/n',
        'state_Tatm2WVatm_xavk': '/state/WV/Tatmxavk',
        'state_Tskin2WVatm_xavk': '/state/WV/Tskinxavk'
    }

    def run(self):
        input = Dataset(self.input().path, 'r')
        with self.output().temporary_path() as target:
            output = Dataset(target, 'w', format=self.format)
            self.copy_dimensions(input, output)
            self.copy_variables(input, output)
            input.close()
            output.close()

    def copy_dimensions(self, input: Dataset, output: Dataset, recursive=True) -> None:
        # find recursively all dimensions of input including subgroups
        if recursive:
            for group in child_groups_of(input):
                target_group = output.createGroup(group.path)
                for name, dim in group.dimensions.items():
                    target_group.createDimension(
                        name, len(dim) if not dim.isunlimited() else None)
        # create only dimensions in root
        else:
            for name, dim in input.dimensions.items():
                output.createDimension(
                    name, len(dim) if not dim.isunlimited() else None)

    def copy_variable(self,  target: Dataset, var: Variable, path: str = None, compressed: bool = False) -> Variable:
        if path:
            out_var = target.createVariable(
                path, var.datatype, var.dimensions, zlib=compressed)
        else:
            out_var = target.createVariable(
                var.name, var.datatype, var.dimensions, zlib=compressed)
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out_var[:] = var[:]

    def copy_variables(self, input: Dataset, output: Dataset, exclusion_pattern=None) -> None:
        input_variables = list(child_variables_of(input))
        counter = 0
        for group, var in input_variables:
            counter += 1
            if group.path == '/':
                path = var.name
            else:
                path = f'{group.path}/{var.name}'
            if exclusion_pattern and re.match(exclusion_pattern, path):
                continue
            message = f'Copying variable {counter} of {len(input_variables)} {path}'
            self.set_status_message(message)
            logger.info(message)
            self.copy_variable(output, var, self.mapping.get(var.name, path))


class MoveVariables(CopyNetcdfFile):
    "Create a copy of netCDF file with variables organized in subgroups"

    def output_directory(self):
        return 'groups'
