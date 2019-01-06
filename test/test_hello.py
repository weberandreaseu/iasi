import unittest
import iasi


class TestHello(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(iasi.say_hello_world('World'), 'Hello World')
