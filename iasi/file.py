import os
import re

import luigi
from luigi import Config
from luigi.util import common_params, inherits, requires
from netCDF4 import Dataset, Variable

from iasi.util import CustomTask


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

    def output(self):
        return self.create_local_target(file=self.file)

    def run(self):
        input = Dataset(self.input().path, 'r')
        output = Dataset(self.output().path, 'w', format=self.format)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        input.close()
        output.close()

    def copy_dimensions(self, input: Dataset, output: Dataset) -> None:
        for name, dim in input.dimensions.items():
            output.createDimension(
                name, len(dim) if not dim.isunlimited() else None)

    def copy_variable(self,  target: Dataset, var: Variable, path: str = None) -> Variable:
        if path:
            out_var = target.createVariable(f'{path}/{var.name}', var.datatype, var.dimensions)
        else:
            out_var = target.createVariable(var.name, var.datatype, var.dimensions)
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out_var[:] = var[:]

    def copy_variables(self, input: Dataset, output: Dataset) -> None:
        for name, var in input.variables.items():
            if self.exclusion_pattern and re.match(self.exclusion_pattern, name):
                continue
            self.copy_variable(output, var)


class MoveVariables(CopyNetcdfFile):

    exclusion_pattern = r"(state)_([A-Z]+\d?)(\S+)"

    def output(self):
        return self.create_local_target('groups', file=self.file)

    def run(self):
        input = Dataset(self.input().path, 'r')
        output = Dataset(self.output().path, 'w', format=self.format)

        variables = input.variables.keys()
        # exclude all variables matching regex
        matches = list(filter(lambda var: re.match(
            self.exclusion_pattern, var), variables))
        self.exclusions = matches
        # get components of matching variables e.g. ('state', 'WV', 'atm_avk')
        var_components = map(lambda var: re.match(
            self.exclusion_pattern, var).groups(), matches)
        # convert them to paths
        paths = map(lambda vars: os.path.join(*vars), var_components)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        # move remaining variables in subdirectories
        for var, path in zip(matches, paths):
            original = input.variables[var]
            out_var = output.createVariable(
                path, original.datatype, original.dimensions)
            out_var.setncatts({k: original.getncattr(k)
                               for k in original.ncattrs()})
            out_var[:] = original[:]

        input.close()
        output.close()
