from iasi.aposteriori import DirectAposteriori, SvdAposteriori, EigenAposteriori
from iasi.composition import Composition, SingularValueComposition, EigenComposition
from iasi.decomposition import Decomposition, SingularValueDecomposition, EigenDecomposition
from iasi.compression import CompressDataset, DecompressDataset
from iasi.file import CopyNetcdfFile, MoveVariables

# configure logging for module
import logging
_log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
_date_format = "%Y-%m-%d %H:%M:%S"
log_formatter = logging.Formatter(_log_format, _date_format)
logging.basicConfig(level=logging.INFO, format=_log_format, datefmt=_date_format)
logging.getLogger('luigi').setLevel(logging.WARNING)