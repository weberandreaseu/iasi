from netCDF4 import Dataset
from iasi import Composition, CompressDataset
from iasi.evaluation import EvaluationCompressionSize
import luigi

task = CompressDataset(
    file='data/input/MOTIV-slice-1000.nc',
    force_upstream=True,
    dst='/tmp/float',
    # force=True
)

if __name__ == "__main__":
    luigi.build([task], local_scheduler=True)
