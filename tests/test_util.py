import unittest

from mpcrl.util import named


class DummyAgent(named.Named):
    ...


class TestNamedAgent(unittest.TestCase):
    def test_init__with_given_name__saves_name_correctly(self):
        dummy = DummyAgent(name="ciao")
        self.assertEqual(dummy.name, "ciao")

    def test_init__without_name__creates_name_with_class(self):
        dummy0 = DummyAgent()
        dummy1 = DummyAgent()
        self.assertEqual(dummy0.name, "DummyAgent0")
        self.assertEqual(dummy1.name, "DummyAgent1")

    def test_str_and_repr(self):
        name = "ciao"
        dummy = DummyAgent(name=name)
        S = dummy.__str__()
        self.assertIn(name, S)
        S = dummy.__repr__()
        self.assertIn(name, S)
        self.assertIn(DummyAgent.__name__, S)


if __name__ == "__main__":
    unittest.main()
