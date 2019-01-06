import unittest
# from iasi import hdo
from iasi import hello

class TestDelta(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(hello.say_hello_world('World'), 'Hello World')