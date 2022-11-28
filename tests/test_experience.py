import os
import tempfile
import unittest
from typing import Tuple

import numpy as np
from csnlp.util import io

from mpcrl.experience import ExperienceReplay

TMPFILENAME: str = ""


class TestExperienceReplay(unittest.TestCase):
    def test_init__initializes_memory_correctly(self):
        items = [object(), object(), object()]
        maxlen = 2
        mem = ExperienceReplay[object](items, maxlen=maxlen, seed=50)
        self.assertEqual(mem.maxlen, maxlen)
        self.assertNotIn(items[0], mem)
        self.assertIn(items[1], mem)
        self.assertIn(items[2], mem)
        self.assertIsInstance(mem.np_random, np.random.Generator)

    def test_experience__is_pickleable(self):
        mem = ExperienceReplay[Tuple[np.ndarray, float]]()
        for _ in range(10):
            mem.append((np.random.rand(10), np.random.rand()))
        self.assertTrue(io.is_pickleable(mem))

        global TMPFILENAME
        TMPFILENAME = next(tempfile._get_candidate_names())
        io.save(TMPFILENAME, ER=mem, check=69)
        mem_: ExperienceReplay = io.load(TMPFILENAME)["ER"]

        for (a, f), (a_, f_) in zip(mem, mem_):
            np.testing.assert_equal(a, a_)
            self.assertEqual(f, f_)

    def tearDown(self) -> None:
        try:
            os.remove(f"{TMPFILENAME}.pkl")
        finally:
            return super().tearDown()


if __name__ == "__main__":
    unittest.main()
