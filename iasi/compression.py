import os
from math import ceil
from typing import Tuple

import luigi
import numpy as np
from netCDF4 import Dataset
from scipy import linalg

from iasi.file import CopyNetcdfFile

class CompressionTask(CopyNetcdfFile):
    dim = luigi.IntParameter()


class SingularValueDecomposition(CompressionTask):

    exclusions = luigi.ListParameter(default=['state_WVatm_avk'])

    def output(self):
        return self.create_local_target('svd', str(self.dim), file=self.file)

    def run(self):
        input = Dataset(self.input().path)
        output = Dataset(self.output().path, 'w', format=self.format)
        output.createDimension('kernel_eigenvalues', self.dim)
        self.copy_dimensions(input, output)
        self.copy_variables(input, output)
        # get relevant dimensions
        events = input.dimensions['event'].size
        grid_levels = input.dimensions['atmospheric_grid_levels'].size
        species = input.dimensions['atmospheric_species'].size
        # create three components
        self.create_variables(output, grid_levels)
        avk = input.variables['state_WVatm_avk'][...]
        nol = input.variables['atm_nol'][...]
        # decompose average kernel for each event
        for event in range(events):
            for row in range(species):
                for column in range(species):
                    levels = nol[event]
                    # TODO: clarify when input is assumed as valid
                    if np.ma.is_masked(levels):
                        continue
                    levels = int(levels)
                    kernel = avk[event, row, column, :levels, :levels]
                    if np.ma.is_masked(kernel):
                        continue
                    if np.isnan(kernel.data).any() or np.isinf(kernel.data).any():
                        continue
                    U, s, Vh = linalg.svd(kernel.data, full_matrices=False)
                    # reduce dimension to given number of eigenvalues
                    # max number of eigenvalues is number of levels
                    dim = min(self.dim, levels)
                    U = U[:, :dim]
                    s = s[:dim]
                    Vh = Vh[:dim, :]
                    output['state_WVatm_avk_U'][event, row,
                                                column, :levels, :dim] = U
                    output['state_WVatm_avk_s'][event, row, column, :dim] = s
                    output['state_WVatm_avk_Vh'][event, row,
                                                 column, :dim, :levels] = Vh
        input.close()
        output.close()

    def create_variables(self, dataset: Dataset, grid_levels: int) -> None:
        U = dataset.createVariable('state_WVatm_avk_U', 'f8',
                                   ('event', 'atmospheric_species', 'atmospheric_species',
                                    'atmospheric_grid_levels', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        U.description = "U component of singular value decomposition"
        s = dataset.createVariable('state_WVatm_avk_s', 'f8',
                                   ('event', 'atmospheric_species',
                                    'atmospheric_species', 'kernel_eigenvalues'),
                                   fill_value=-9999.9)
        s.description = "Eigenvalues (sigma) of singular value decomposition"
        Vh = dataset.createVariable('state_WVatm_avk_Vh', 'f8',
                                    ('event', 'atmospheric_species', 'atmospheric_species',
                                     'kernel_eigenvalues', 'atmospheric_grid_levels'),
                                    fill_value=-9999.9)
        Vh.description = "Vh component of singular value decomposition"
