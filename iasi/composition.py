from netCDF4 import Dataset, Group, Variable

from iasi.file import CopyNetcdfFile, MoveVariables
from luigi.util import requires


@requires(MoveVariables)
class Compositon(CopyNetcdfFile):
    pass


class SingularValueComposition:

    def __init__(self, group: Group):
        # assert group contains U, s, VH
        pass


class EigenCompositon:
    
    def __init__(self, group: Group):
        # assert group contains Q, s
        pass
