import logging
import os
from typing import Dict, List, Tuple

import luigi
import numpy as np
from luigi.util import inherits, requires
from netCDF4 import Dataset, Group, Variable

import iasi

logger = logging.getLogger(__name__)


def dimensions_of(variable: Variable) -> Dict:
    return {name: dim for name, dim in zip(variable.dimensions, variable.shape)}


def child_groups_of(group: Group):
    yield group
    if group.groups:
        for subgroup in group.groups.values():
            yield from child_groups_of(subgroup)


def child_variables_of(group: Group):
    for subgroup in child_groups_of(group):
        for variable in subgroup.variables.values():
            yield (subgroup, variable)


def root_group_of(group: Group) -> Group:
    if group.parent:
        return root_group_of(group.parent)
    else:
        return group


class custom(luigi.Config):
    tracking_url = luigi.Parameter(default='http://localhost:8082')


class CustomTask(luigi.Task):
    """Base luigi task which provides common attributes accessable by subclasses

    Attributes:
        force           remove output and run again
        force_upstream  recursively remove upstream input and run again
    """
    force = luigi.BoolParameter(significant=False, default=False)
    force_upstream = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        # source https://github.com/spotify/luigi/issues/595#issuecomment-314127254
        super().__init__(*args, **kwargs)
        if self.force_upstream is True:
            self.force = True
        if self.force is True:
            done = False
            tasks = [self]
            while not done:
                if not issubclass(tasks[0].__class__, luigi.ExternalTask):
                    # do not delete output of external tasks
                    outputs = luigi.task.flatten(tasks[0].output())
                    [os.remove(out.path) for out in outputs if out.exists()]
                if self.force_upstream is True:
                    tasks += luigi.task.flatten(tasks[0].requires())
                tasks.pop(0)
                if len(tasks) == 0:
                    done = True


    @luigi.Task.event_handler(luigi.Event.START)
    def callback_start(self):
        if hasattr(self, 'set_tracking_url') and callable(self.set_tracking_url):
            self.set_tracking_url(custom().tracking_url)
        logger.info('Starting Task %s...', type(self).__name__)

    def log_file(self):
        raise NotImplementedError

    def create_local_path(self, *args: str, file: str, ext: str = None) -> str:
        _, filename = os.path.split(file)
        if ext:
            name, _ = os.path.splitext(filename)
            filename = f'{name}.{ext}'
        path = os.path.join(self.dst, *args)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    def create_local_target(self, *args: str, file: str, ext: str = None) -> luigi.LocalTarget:
        _, filename = os.path.split(file)
        if ext:
            name, _ = os.path.splitext(filename)
            filename = f'{name}.{ext}'
        path = os.path.join(self.dst, *args, filename)
        target = luigi.LocalTarget(path)
        target.makedirs()
        return target
