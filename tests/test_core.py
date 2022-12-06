import unittest
from typing import Optional, Tuple, Union

import numpy as np
from parameterized import parameterized

from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import (
    EpsilonGreedyExploration,
    GreedyExploration,
    NoExploration,
)
from mpcrl.core.random import np_random


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

    def test_sample__raises__with_no_maxlen_and_percentage_size(self):
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=None)
        with self.assertRaises(TypeError):
            list(mem.sample(n=0.0))

    @parameterized.expand([(0,), (float(0),)])
    def test_sample__with_zero_samples__returns_no_samples(self, n: Union[int, float]):
        mem = ExperienceReplay[Tuple[np.ndarray, float]](maxlen=100)
        self.assertListEqual(list(mem.sample(n)), [])

    @parameterized.expand([(10,), (0.1,)])
    def test_sample__returns_right_sample_size(self, n: Union[int, float]):
        N = 100
        Nsample = 10
        mem = ExperienceReplay[int](maxlen=N)
        mem.extend(range(N))
        sample = list(mem.sample(n=n))
        self.assertEqual(len(sample), Nsample)
        for item in sample:
            self.assertIn(item, mem)

    @parameterized.expand([(20, 10), (20, 0.5), (0.2, 10), (0.2, 0.5)])
    def test_sample__last_n__includes_last_n_items(
        self, n: Union[int, float], last_n: Union[int, float]
    ):
        N = 100
        Nsample = 20
        Nlast = 10
        mem = ExperienceReplay[int](maxlen=N)
        mem.extend(range(N))
        sample = list(mem.sample(n=n, last_n=last_n))
        self.assertEqual(len(sample), Nsample)
        for item in sample:
            self.assertIn(item, mem)
        for item in range(N - Nlast, N):
            self.assertIn(item, sample)


class TestRandom(unittest.TestCase):
    def test_np_random__raises__with_invalid_seed(self):
        with self.assertRaisesRegex(
            ValueError, "Seed must be a non-negative integer or omitted, not -1."
        ):
            np_random(-1)

    @parameterized.expand([(69,), (None,)])
    def test_np_random__initializes_rng_with_correct_seed(self, seed: Optional[int]):
        rng, actual_seed = np_random(seed)
        self.assertIsInstance(rng, np.random.Generator)
        if seed is not None:
            self.assertEqual(seed, actual_seed)
        else:
            self.assertIsInstance(actual_seed, int)


class TestExploration(unittest.TestCase):
    def test_no_exploration__never_explores(self):
        exploration = NoExploration()
        self.assertFalse(exploration.can_explore())
        np.testing.assert_equal(exploration.perturbation(), np.nan)
        exploration.decay()  # does nothing

    def test_greedy_exploration__instantiates_np_random(self):
        exploration = GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        self.assertIsInstance(exploration.np_random, np.random.Generator)

    def test_greedy_exploration__always_explores(self):
        exploration = GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        self.assertTrue(exploration.can_explore())

    def test_greedy_exploration__decays_strength(self):
        exploration = GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        for _ in range(5):
            exploration.decay()
        np.testing.assert_allclose(exploration.strength, 0.5 * 0.75**5)

    def test_greedy_exploration__perturbs(self):
        exploration = GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        for method in ("random", "uniform"):
            exploration.perturbation(method)

    def test_epsilon_greedy_exploration__init(self):
        exploration = EpsilonGreedyExploration(
            epsilon=0.7, strength=0.5, epsilon_decay_rate=0.75
        )
        self.assertEqual(exploration.strength_decay_rate, 0.75)

    def test_epsilon_greedy_exploration__sometimes_explores(self):
        exploration = EpsilonGreedyExploration(
            epsilon=0.7, strength=0.5, epsilon_decay_rate=0.75, seed=42
        )
        self.assertTrue(exploration.can_explore())
        found_true, found_false = False, False
        for _ in range(100):
            explore = exploration.can_explore()
            if explore:
                found_true = True
            else:
                found_false = True
            if found_true and found_false:
                break
        self.assertTrue(found_true and found_false)

    def test_epsilon_greedy_exploration__decays_strength(self):
        exploration = EpsilonGreedyExploration(
            epsilon=0.7, strength=0.5, epsilon_decay_rate=0.75, strength_decay_rate=0.2
        )
        for _ in range(5):
            exploration.decay()
        np.testing.assert_allclose(exploration.epsilon, 0.7 * 0.75**5)
        np.testing.assert_allclose(exploration.strength, 0.5 * 0.2**5)


if __name__ == "__main__":
    unittest.main()
