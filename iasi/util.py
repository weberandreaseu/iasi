import logging
import os
from typing import List

import luigi
import numpy as np
from luigi.util import inherits, requires
from netCDF4 import Dataset, Group, Variable


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


def child_groups_of(group: Group):
    yield group
    if group.groups:
        for subgroup in group.groups.values():
            yield from child_groups_of(subgroup)


def child_variables_of(group: Group):
    for subgroup in child_groups_of(group):
        for variable in subgroup.variables.values():
            yield (subgroup, variable)


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
