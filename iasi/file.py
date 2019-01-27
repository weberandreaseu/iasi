import luigi
from iasi.util import CustomTask
from netCDF4 import Dataset
import os


class ReadFile(luigi.ExternalTask):
    """Basic class for reading a local file as input for a luigi task.

    Attributes:
        file    path to local file to open
    """
    file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.file)


class CopyNetcdfFile(CustomTask):
    """Luigi Task for copying netCDF files with a subset of variables

    Attributes:  
        file        path to local file to open
        inclusions  variables to include
        exclusions  variables to exclude
    """
    file = luigi.Parameter()
    inclusions = luigi.ListParameter(default=[])
    exclusions = luigi.ListParameter(default=[])
    format = luigi.Parameter(default='NETCDF4')

    def requires(self):
        if self.exclusions and self.inclusions:
            raise AttributeError('Only inclusions OR exclusions are allowed.')
        return ReadFile(file=self.file)

    def output(self):
        _, file = os.path.split(self.file)
        path = os.path.join(self.dst, file)
        target = luigi.LocalTarget(path)
        target.makedirs()
        return target

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
