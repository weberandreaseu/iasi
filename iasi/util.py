import logging
import os
from typing import List, Dict

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
        dst             base directory for output
        log             write task logs to file
        force           remove output and run again
        force_upstream  recursively remove upstream input and run again
    """
    dst = luigi.Parameter()
    log = luigi.BoolParameter(significant=False, default=True)
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

    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def callback_execution_time(self, execution_time):
        logger.info('Task %s executeted in %s seconds', type(
            self).__name__, str(execution_time))

    @luigi.Task.event_handler(luigi.Event.START)
    def callback_start(self):
        if hasattr(self, 'set_tracking_url') and callable(self.set_tracking_url):
            self.set_tracking_url(custom().tracking_url)
        if self.log:
            try:
                # log file destination has to be implemented by concrete task
                self.log_handler = logging.FileHandler(
                    self.log_file(), mode='w')
                self.log_handler.setFormatter(iasi.log_formatter)
                logging.getLogger().addHandler(self.log_handler)
            except NotImplementedError:
                pass
        logger.info('Starting Task %s...', type(self).__name__)

    @luigi.Task.event_handler(luigi.Event.SUCCESS)
    def callback_success(self):
        logger.info('Task %s successfully finished', type(self).__name__)
        self._remove_log_hander()

    @luigi.Task.event_handler(luigi.Event.FAILURE)
    def callback_failure(self, error):
        logger.error('Task %s failed', type(self).__name__)
        logger.error('Message: %s', error)
        self._remove_log_hander()

    def _remove_log_hander(self):
        if hasattr(self, 'log_handler'):
            self.log_handler.close()
            logging.getLogger().removeHandler(self.log_handler)

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
