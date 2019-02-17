#!/usr/bin/env python
from netCDF4 import Dataset
import sys

assert len(sys.argv) == 4

slice = int(sys.argv[1])
input = sys.argv[2]
output = sys.argv[3]


def copy_dimensions(input: Dataset, output: Dataset) -> None:
    for name, dim in input.dimensions.items():
        output.createDimension(
            name, len(dim) if not dim.isunlimited() else None)


def copy_variables(input: Dataset, output: Dataset, slice) -> None:
    # source https://gist.github.com/guziy/8543562
    for name, var in input.variables.items():
        out_var = output.createVariable(name, var.datatype, var.dimensions)
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        if var.dimensions[0] == 'event':
            out_var[:slice] = var[:slice]
        else:
            out_var[:] = var[:]


with Dataset(input) as input, Dataset(output, 'w') as output:
    copy_dimensions(input, output)
    copy_variables(input, output, slice)
