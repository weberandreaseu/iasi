import os
import re

import luigi
from luigi import Config
from luigi.util import common_params, inherits, requires
# from iasi.compression import CompressDataset
from netCDF4 import Dataset, Variable, Group

from iasi.util import CustomTask, child_variables_of
import logging


class ReadFile(luigi.ExternalTask):
    """Basic class for reading a local file as input for a luigi task."""
    file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.file)


@requires(ReadFile)
class CopyNetcdfFile(CustomTask):
    """Luigi Task for copying netCDF files with a subset of variables

    Attributes:
        exclusion_pattern   variables matching pattern are excluded
        format              used netCDF format
    """
    exclusion_pattern = luigi.Parameter(default=None)
    format = luigi.Parameter(default='NETCDF4')

    mapping = {
        'cld_cov'                     :'/cld/cov',
        'cld_tp'                      :'/cld/tp',
        'cld_ph'                      :'/cld/ph',
        'cld_tt'                      :'/cld/tt',
        # TODO typo?
        'flg_cdlfrm'                  :'/flg/cldfrm',
        'flg_cldnes'                  :'/flg/cldnes',
        'flg_dustcld'                 :'/flg/dustcld',
        'flg_iasicld'                 :'/flg/iasicld',
        'flg_initia'                  :'/flg/initia',
        'flg_itconv'                  :'/flg/itconv',
        'flg_numit'                   :'/flg/numit',
        'flg_resid'                   :'/flg/resid',
        'flg_retcheck'                :'/flg/retcheck',
        # TODO is this desired?
        'srf_flag'                    :'/flg/srf',
        'sfc_emi'                     :'/sfc/emi',
        'sfc_emi_wn'                  :'/sfc/emi_wn',
        'sfc_emi_flag'                :'/sfc/emi_flag',
        'state_GHGatm'                :'/state/GHG/r',
        'state_GHGatm_a'              :'/state/GHG/a',
        'state_GHGatm_avk'            :'/state/GHG/avk',
        'state_GHGatm_n'              :'/state/GHG/n',
        'state_Tatm2GHGatm_xavk'      :'/state/GHG/Tatmxavk',
        'state_Tskin2GHGatm_xavk'     :'/state/GHG/Tskinxavk',
        'state_HNO3atm'               :'/state/HNO3/r',
        'state_HNO3atm_a'             :'/state/HNO3/a',
        'state_HNO3atm_avk'           :'/state/HNO3/avk',
        'state_HNO3atm_n'             :'/state/HNO3/n',
        'state_Tatm2HNO3atm_xavk'     :'/state/HNO3/Tatmxavk',
        'state_Tskin2HNO3atm_xavk'    :'/state/HNO3/Tskinxavk',
        'state_Tatm'                  :'/state/Tatm/r',
        'state_Tatm_a'                :'/state/Tatm/a',
        'state_Tatm_avk'              :'/state/Tatm/avk',
        'state_Tatm_n'                :'/state/Tatm/n',
        'state_Tskin2Tatm_xavk'       :'/state/Tatm/Tskinxavk',
        'state_Tskin'                 :'/state/Tskin/r',
        'state_Tskin_a'               :'/state/Tskin/a',
        'state_Tskin_n'               :'/state/Tskin/n',
        'state_WVatm'                 :'/state/WV/r',
        'state_WVatm_a'               :'/state/WV/a',
        'state_WVatm_avk'             :'/state/WV/avk',
        'state_WVatm_n'               :'/state/WV/n',
        'state_Tatm2WVatm_xavk'       :'/state/WV/Tatmxavk',
        'state_Tskin2WVatm_xavk'      :'/state/WV/Tskinxavk'
    }

    def output(self):
        return self.create_local_target(file=self.file)

    def run(self):
        input = Dataset(self.input().path, 'r')
        try:
            output = Dataset(self.output().path, 'w', format=self.format)
        except PermissionError as pe:
            logging.error('Permission error. File %s probably exists already', self.output().path)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        input.close()
        output.close()

    def copy_dimensions(self, input: Dataset, output: Dataset) -> None:
        for name, dim in input.dimensions.items():
            output.createDimension(name, len(dim) if not dim.isunlimited() else None)

    def copy_variable(self,  target: Dataset, var: Variable, path: str = None, compressed: bool = False) -> Variable:
        if path:
            out_var = target.createVariable(path, var.datatype, var.dimensions, zlib=compressed)
        else:
            out_var = target.createVariable(var.name, var.datatype, var.dimensions, zlib=compressed)
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out_var[:] = var[:]

    def copy_variables(self, input: Dataset, output: Dataset) -> None:
        for group, var in child_variables_of(input):
            if group.path == '/':
                path = var.name
            else:
                path = f'{group.path}/{var.name}'
            if self.exclusion_pattern and re.match(self.exclusion_pattern, path):
                continue
            self.copy_variable(output, var, self.mapping.get(var.name, path))


class MoveVariables(CopyNetcdfFile):
    "Create a copy of netCDF file with variables organized in subgroups"
    def output(self):
        return self.create_local_target('groups', file=self.file)
