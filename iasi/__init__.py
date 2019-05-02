# configure logging for module
import logging as _logging

from iasi.composition import Composition
from iasi.compression import CompressDataset, DecompressDataset

_log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
_date_format = "%Y-%m-%d %H:%M:%S"
log_formatter = _logging.Formatter(_log_format, _date_format)
_logging.basicConfig(level=_logging.INFO, format=_log_format,
                    datefmt=_date_format)
_logging.getLogger('luigi').setLevel(_logging.WARNING)

# logger = logging.getLogger(__name__)

# class _NumpyLogging:

#     def write(self, msg):
#         logger.warn(msg)


# np.seterrcall(_NumpyLogging())
# np.seterr(all='log')
