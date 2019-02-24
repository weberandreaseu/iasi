import os
import unittest

from iasi.util import CustomTask
from iasi import GroupCompression, MoveVariables
import luigi
from netCDF4 import Dataset, Variable, Group
from iasi.util import Quadrant, TwoQuadrants, FourQuadrants, child_variables_of, child_groups_of
import numpy as np


class TestCustomTask(unittest.TestCase):
    def test_create_local_target(self):
        task = CustomTask(dst='/tmp/iasi')
        target = task.create_local_target(
            'custom_target', '28', file='dummy.nc', ext='csv')
        self.assertTrue(os.path.exists('/tmp/iasi/custom_target/28/'))
        self.assertEqual(target.path, '/tmp/iasi/custom_target/28/dummy.csv')


class TestQuadrants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = 'test/resources/MOTIV-single-event.nc'
        # make sure there is a compressed file for testing purpose
        compression = GroupCompression(
            file=file,
            dst='/tmp/iasi'
        )
        uncompressed = MoveVariables(
            file=file,
            dst='/tmp/iasi'
        )
        luigi.build([compression, uncompressed], local_scheduler=True)
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
        self.assertListEqual(names, ['state', 'GHG',  'T', 'HNO3', 'WV'])

    def test_variables_of_group(self):
        wv = self.uncompressed['state/WV']
        self.assertIsInstance(wv, Group)
        children = [var.name for g, var in child_variables_of(wv)]
        self.assertListEqual(children, ['atm', 'atm_a', 'atm_n', 'atm_avk'])

    def test_single_quadrant_assembly(self):
        avk = self.uncompressed['/state/HNO3/atm_avk']
        q: Quadrant = Quadrant.for_assembly(avk)
        self.assertTupleEqual(q.assembly_shape(), (1, 29, 29))
        array = np.random.uniform(size=(29, 29))
        assembly = q.assemble(array, 23)
        self.assertTupleEqual(assembly.shape, (23, 23))
        self.assertTrue(np.allclose(array[: 23, : 23], assembly))

    def test_two_quadrants_assembly(self):
        xavk = self.uncompressed['/state/T/atm2GHGatm_xavk']
        q: Quadrant = Quadrant.for_assembly(xavk)
        self.assertIsInstance(q, TwoQuadrants)
        self.assertTupleEqual(q.assembly_shape(), (1, 29, 58))
        array = np.random.uniform(size=(2, 29, 29))
        assembly = q.assemble(array, 23)
        self.assertTupleEqual(assembly.shape, (23, 46))

    def test_four_quadrants_assembly(self):
        avk = self.uncompressed['/state/WV/atm_avk']
        q: Quadrant = Quadrant.for_assembly(avk)
        self.assertIsInstance(q, FourQuadrants)
        self.assertTupleEqual(q.assembly_shape(), (1, 58, 58))
        array = np.random.uniform(size=(2, 2, 29, 29))
        assembly = q.assemble(array, 23)
        self.assertTupleEqual(assembly.shape, (46, 46))

    def test_single_quadrant_disassembly(self):
        atm_n = self.compressed['state/HNO3/atm_n/Q']
        q: Quadrant = Quadrant.for_disassembly(atm_n)
        self.assertIsInstance(q, Quadrant)
        self.assertTupleEqual(q.disassembly_shape(), (1, 29, 29))
        array = np.random.uniform(size=(29, 29))
        disassembly = q.disassemble(array, 23)
        self.assertTupleEqual(disassembly.shape, (23, 23))

    def test_two_quadrant_disassembly(self):
        xavk = self.compressed['state/T/atm2GHGatm_xavk/Vh']
        q: Quadrant = Quadrant.for_disassembly(xavk)
        self.assertIsInstance(q, TwoQuadrants)
        self.assertTupleEqual(q.disassembly_shape(), (1, 2, 29, 29))
        array = np.arange(29*58).reshape(29, 58)
        disassembly = q.disassemble(array, 23)
        self.assertTupleEqual(disassembly.shape, (2, 23, 23))
        close = np.allclose(array[:23, 29:52], disassembly[1, :23,:23])
        self.assertTrue(close)

    def test_four_quadrant_disassembly(self):
        avk = self.compressed['state/WV/atm_avk/U']
        q: Quadrant = Quadrant.for_disassembly(avk)
        self.assertIsInstance(q, FourQuadrants)
        self.assertTupleEqual(q.disassembly_shape(), (1, 2, 2, 29, 29))
        array = np.arange(58*58).reshape(58, 58)
        disassembly = q.disassemble(array, 23)
        self.assertTupleEqual(disassembly.shape, (2, 2, 23, 23))
        close = np.allclose(array[29:52, 29:52], disassembly[1, 1, :23, :23])
        self.assertTrue(close)
