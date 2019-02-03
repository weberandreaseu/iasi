import luigi
from iasi.util import CustomTask
from netCDF4 import Dataset
import os
from luigi.util import inherits, requires, common_params
from luigi import Config


class ReadFile(luigi.ExternalTask):
    """Basic class for reading a local file as input for a luigi task."""
    file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.file)


@requires(ReadFile)
class CopyNetcdfFile(CustomTask):
    """Luigi Task for copying netCDF files with a subset of variables

    Attributes:  
        inclusions  variables to include
        exclusions  variables to exclude
        format      used netCDF format
    """
    inclusions = luigi.ListParameter(default=[])
    exclusions = luigi.ListParameter(default=[])
    format = luigi.Parameter(default='NETCDF4')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exclusions and self.inclusions:
            raise AttributeError('Only inclusions OR exclusions are allowed.')

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

    def copy_variables(self, input: Dataset, output: Dataset) -> None:
        # source https://gist.github.com/guziy/8543562
        for name, var in input.variables.items():
            if name in self.exclusions or (self.inclusions and name not in self.inclusions):
                continue
            out_var = output.createVariable(name, var.datatype, var.dimensions)
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            out_var[:] = var[:]
