import unittest

import luigi
import numpy as np
from netCDF4 import Dataset, Group, Variable

from iasi import CompressDataset, MoveVariables
from iasi.quadrant import Quadrant, AssembleTwoQuadrants, AssembleFourQuadrants, DisassembleTwoQuadrants, DisassembleFourQuadrants
from iasi.util import child_groups_of, child_variables_of


class TestQuadrants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = CompressDataset(
            file=file,
            dst='/tmp/iasi',
            force=True,
            log=False
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi',
            force=True,
            log=False
        )
        assert luigi.build([compression, uncompressed], local_scheduler=True)
        cls.compressed = Dataset(compression.output().path)
        cls.uncompressed = Dataset(uncompressed.output().path)

    @classmethod
    def tearDownClass(cls):
        cls.compressed.close()
        cls.uncompressed.close()

    def test_child_groups(self):
        state = self.uncompressed['state']
        children = child_groups_of(state)
        names = [g.name for g in children]
        self.assertListEqual(
            names, ['state', 'GHG', 'HNO3', 'Tatm', 'Tskin', 'WV'])

    def test_variables_of_group(self):
        wv = self.uncompressed['state/WV']
        self.assertIsInstance(wv, Group)
        children = [var.name for g, var in child_variables_of(wv)]
        self.assertListEqual(
            children, ['r', 'a', 'n', 'avk', 'Tatmxavk', 'Tskinxavk'])

    def test_single_quadrant_assembly(self):
        avk = self.uncompressed['state/HNO3/avk']
        q: Quadrant = Quadrant.for_assembly('HNO3', 'avk', avk)
        self.assertIsInstance(q, Quadrant)
        self.assertTupleEqual(q.transformed_shape(), (1, 29, 29))
        array = np.random.uniform(size=(29, 29))
        assembly = q.transform(array, 23)
        self.assertTupleEqual(assembly.shape, (23, 23))
        self.assertTrue(np.allclose(array[: 23, : 23], assembly))

    def test_two_quadrants_assembly(self):
        xavk = self.uncompressed['state/GHG/Tatmxavk']
        q: Quadrant = Quadrant.for_assembly('GHG', xavk.name, xavk)
        self.assertIsInstance(q, AssembleTwoQuadrants)
        self.assertTupleEqual(q.transformed_shape(), (1, 58, 29))
        array = np.random.uniform(size=(2, 29, 29))
        assembly = q.transform(array, 23)
        self.assertTupleEqual(assembly.shape, (46, 23))

    def test_four_quadrants_assembly(self):
        avk = self.uncompressed['/state/WV/avk']
        q: Quadrant = Quadrant.for_assembly('WV', 'avk', avk)
        self.assertIsInstance(q, AssembleFourQuadrants)
        self.assertTupleEqual(q.transformed_shape(), (1, 58, 58))
        array = np.random.uniform(size=(2, 2, 29, 29))
        assembly = q.transform(array, 23)
        self.assertTupleEqual(assembly.shape, (46, 46))
        close = np.allclose(assembly[23:23*2, :23], array[0, 1, :23, :23])
        self.assertTrue(close, 'Four quadrant assembly not close')

    def test_single_quadrant_disassembly(self):
        atm_n = self.compressed['state/HNO3/n/Q']
        q: Quadrant = Quadrant.for_disassembly('HNO3', 'n', atm_n)
        self.assertIsInstance(q, Quadrant)
        self.assertTupleEqual(q.transformed_shape(), (1, 29, 29))
        array = np.random.uniform(size=(29, 29))
        disassembly = q.transform(array, 23)
        self.assertTupleEqual(disassembly.shape, (23, 23))

    def test_two_quadrant_disassembly(self):
        xavk = self.compressed['state/GHG/Tatmxavk/U']
        q: Quadrant = Quadrant.for_disassembly('GHG', 'Tatmxavk', xavk)
        self.assertIsInstance(q, DisassembleTwoQuadrants)
        self.assertTupleEqual(q.transformed_shape(), (1, 2, 29, 29))
        array = np.arange(29*58).reshape(58, 29)
        disassembly = q.transform(array, 23)
        self.assertTupleEqual(disassembly.shape, (2, 23, 23))
        close = np.allclose(array[:23, :23], disassembly[0, :23, :23])
        self.assertTrue(close)
        close = np.allclose(array[29:52, :23], disassembly[1, :23, :23])
        self.assertTrue(close)

    def test_four_quadrant_disassembly(self):
        avk_rc = self.compressed['state/WV/avk/U']
        q: Quadrant = Quadrant.for_disassembly('WV', 'avk', avk_rc)

        self.assertIsInstance(q, DisassembleFourQuadrants)
        self.assertTupleEqual(q.transformed_shape(), (1, 2, 2, 29, 29))
        avk = self.uncompressed['state/WV/avk/']
        q_assembly = Quadrant.for_assembly('WV', 'avk', avk)
        array = np.arange(58*58).reshape(58, 58)
        disassembly = q.transform(array, 23)
        array_rc = q_assembly.transform(disassembly, 23)
        self.assertTupleEqual(disassembly.shape, (2, 2, 23, 23))
        close = np.allclose(array[29:52, 29:52], disassembly[1, 1, :23, :23])
        self.assertTrue(close)
        for i in range(2):
            for j in range(2):
                close = np.allclose(
                    array[i * 29:23 + i * 29, j * 29:23 + j * 29],
                    array_rc[i * 23:(i+1) * 23, j * 23:(j+1) * 23]
                )
                self.assertTrue(close)
