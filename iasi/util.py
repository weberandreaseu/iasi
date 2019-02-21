import os

import luigi
from luigi.util import inherits, requires
from netCDF4 import Dataset, Variable
import numpy as np
from typing import List

import logging


class CommonParams(luigi.Task):
    dst = luigi.Parameter()


class ForceableTask(luigi.Task):
    # source https://github.com/spotify/luigi/issues/595#issuecomment-314127254
    force = luigi.BoolParameter(significant=False, default=False)
    force_upstream = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.force_upstream is True:
            self.force = True
        if self.force is True:
            done = False
            tasks = [self]
            while not done:
                outputs = luigi.task.flatten(tasks[0].output())
                [os.remove(out.path) for out in outputs if out.exists()]
                if self.force_upstream is True:
                    tasks += luigi.task.flatten(tasks[0].requires())
                tasks.pop(0)
                if len(tasks) == 0:
                    done = True


class CustomTask(ForceableTask):
    """Base luigi task which provides common attributes accessable by subclasses

    Attributes:
        file    path to local file to open
        dst     base directory for output
    """
    dst = luigi.Parameter()

    def create_local_target(self, *args: str, file: str, ext: str = None) -> luigi.LocalTarget:
        _, filename = os.path.split(file)
        if ext:
            name, _ = os.path.splitext(filename)
            filename = f'{name}.{ext}'
        path = os.path.join(self.dst, *args, filename)
        target = luigi.LocalTarget(path)
        target.makedirs()
        return target


class Quadrant:

    assembles = ('event', 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = assembles

    @classmethod
    def for_assembly(cls, variable: Variable):
        # get and initialize quadrant which assembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.assembles == dimensions, [cls, TwoQuadrants, FourQuadrants]))(variable)

    @classmethod
    def for_disassembly(cls, variable: Variable):
        # get and initialize quadrant which disassembles the given dimensions
        dimensions = variable.dimensions
        return next(filter(lambda q: q.disassembles == dimensions, [cls, TwoQuadrants, FourQuadrants]))(variable)

    def __init__(self, variable: Variable):
        self.var = variable

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def disassemble(self, array: np.ma.MaskedArray, levels: int):
        return array[:levels, :levels]

    def create_variable(self, output: Dataset):
        raise NotImplementedError

    def assembly_shape(self):
        return self.var.shape

    def disassembly_shape(self):
        return self.var.shape


class TwoQuadrants(Quadrant):

    assembles = ('event', 'atmospheric_species',
                 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = ('event', 'atmospheric_grid_levels',
                    'double_atmospheric_grid_levels')

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return np.block([array[0, :levels, :levels], array[1, :levels, :levels]])

    def assembly_shape(self):
        grid_levels = self.var.shape[3]
        return (self.var.shape[0], grid_levels, grid_levels * 2)

    def disassembly_shape(self):
        grid_levels = self.var.shape[1]
        return (self.var.shape[0], 2, grid_levels, grid_levels)

    def disassemble(self, array: np.ma.MaskedArray, levels: int):
        reshape = self.disassembly_shape()[1:]
        return np.reshape(array, reshape)[:, :levels, :levels]


class FourQuadrants(Quadrant):

    assembles = ('event', 'atmospheric_species', 'atmospheric_species',
                 'atmospheric_grid_levels', 'atmospheric_grid_levels')
    disassembles = ('event', 'double_atmospheric_grid_levels',
                    'double_atmospheric_grid_levels')

    def assemble(self, array: np.ma.MaskedArray, levels: int):
        return np.block([
            [array[0, 0, :levels, :levels], array[0, 1, :levels, :levels]],
            [array[1, 0, :levels, :levels], array[1, 1, :levels, :levels]]
        ])

    def assembly_shape(self):
        grid_levels = self.var.shape[4]
        return (self.var.shape[0], grid_levels * 2, grid_levels * 2)

    def disassembly_shape(self):
        grid_levels = self.var.shape[2]
        return (self.var.shape[0], 2, 2, int(grid_levels / 2), int(grid_levels / 2))

    def disassemble(self, array: np.ma.MaskedArray, levels: int):
        reshape = self.disassembly_shape()[1:]
        return np.reshape(array, reshape)[:, :, :levels, :levels]
