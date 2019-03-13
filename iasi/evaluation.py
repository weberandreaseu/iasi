import luigi
import pandas as pd
import os

from iasi.compression import CompressDataset
from iasi.file import MoveVariables
from iasi.util import CustomTask


class EvaluateCompression(CustomTask):

    file = luigi.Parameter()

    def requires(self):
        # compression levels
        return {
            'original': MoveVariables(file=self.file, dst=self.dst),
            'reconstructed': [CompressDataset(
                file=self.file,
                dst=self.dst,
                thres_eigenvalues=thres,
                force=self.force
            ) for thres in [1e-2, 1e-3, 1e-4, 1e-5]]
        }

    def run(self):
        df = pd.DataFrame()
        df = df.append({
            'compressed': False,
            'size': self.size_in_kb(self.input()['original'].path)
        }, ignore_index=True)
        for task, input in zip(self.requires()['reconstructed'], self.input()['reconstructed']):
            df = df.append({
                'compressed': True,
                'size': self.size_in_kb(input.path),
                'thres_eigenvalues': task.thres_eigenvalues
            }, ignore_index=True)
        print('\n', df)
        df.to_csv(self.output().path)

    def size_in_kb(self, file):
        return int(os.path.getsize(file) / 1000)

    def output(self):
        return self.create_local_target('compression-summary', file=self.file, ext='csv')
