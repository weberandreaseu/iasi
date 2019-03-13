import os

import luigi
import pandas as pd
from sklearn.model_selection import ParameterGrid

from iasi.compression import CompressDataset, SelectSingleVariable
from iasi.file import MoveVariables
from iasi.util import CustomTask


class EvaluateCompression(CustomTask):

    file = luigi.Parameter()
    variable = luigi.Parameter()

    def requires(self):
        compression_parameter = {
            'compressed': [True],
            'file': [self.file],
            'dst': [self.dst],
            'force': [self.force],
            'threshold': [1e-2, 1e-3, 1e-4, 1e-5],
            'variable': [self.variable]
        }
        parameter_grid = list(ParameterGrid(compression_parameter))
        # add parameter for 
        parameter_grid.append({
            'compressed': False,
            'file': self.file,
            'dst': self.dst,
            'force': self.force,
            'variable': self.variable
        })
        return [SelectSingleVariable(**params) for params in parameter_grid]

    def run(self):
        df = pd.DataFrame()
        for task, input in zip(self.requires(), self.input()):
            df = df.append({
                'variable': task.variable,
                'compressed': task.compressed,      
                'size': self.size_in_kb(input.path),
                'threshold': task.threshold
                # 'file': task.file,
            }, ignore_index=True)
        print('\n', df)
        df.to_csv(self.output().path)

    def size_in_kb(self, file):
        return int(os.path.getsize(file) / 1000)

    def output(self):
        return self.create_local_target('compression-summary', file=self.file, ext='csv')
