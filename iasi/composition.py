from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from luigi.util import requires


@requires(MoveVariables)
class Compositon(CopyNetcdfFile):
    pass
