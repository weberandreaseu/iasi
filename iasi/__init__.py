from iasi.aposteriori import DirectAposteriori, SvdAposteriori, EigenAposteriori
from iasi.composition import Composition, SingularValueComposition, EigenComposition
from iasi.decomposition import Decomposition, SingularValueDecomposition, EigenDecomposition
from iasi.compression import CompressDataset, DecompressDataset
from iasi.file import CopyNetcdfFile, MoveVariables

# configure logging for module
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('luigi').setLevel(logging.WARNING)