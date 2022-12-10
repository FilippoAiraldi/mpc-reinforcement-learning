import pickle
import unittest
from itertools import product
from typing import Iterable, Optional, Tuple, Union

import casadi as cs
import numpy as np
from parameterized import parameterized

from mpcrl import ExperienceReplay, LearnableParameter, LearnableParametersDict
from mpcrl import exploration as E
from mpcrl.core.random import make_seeds, np_random


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

    @parameterized.expand([(5,), (None,), (range(100),)])
    def test_make_seeds__returns_expected(self, seed: Union[None, int, Iterable[int]]):
        N = 100
        if seed is None:
            expected_seeds = [None] * N
        elif isinstance(seed, int):
            expected_seeds = [seed + i for i in range(N)]
        else:
            expected_seeds = seed
        seeds1, seeds2 = list(zip(*zip(expected_seeds, make_seeds(seed))))
        self.assertListEqual(list(seeds1), list(seeds2))


class TestExploration(unittest.TestCase):
    def test_no_exploration__never_explores(self):
        exploration = E.NoExploration()
        self.assertFalse(exploration.can_explore())
        np.testing.assert_equal(exploration.perturbation(), np.nan)
        exploration.decay()  # does nothing

    def test_greedy_exploration__instantiates_np_random(self):
        exploration = E.GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        self.assertIsInstance(exploration.np_random, np.random.Generator)

    def test_greedy_exploration__always_explores(self):
        exploration = E.GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        self.assertTrue(exploration.can_explore())

    def test_greedy_exploration__decays_strength(self):
        exploration = E.GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        for _ in range(5):
            exploration.decay()
        np.testing.assert_allclose(exploration.strength, 0.5 * 0.75**5)

    @parameterized.expand([("uniform",), ("normal",), ("standard_normal",)])
    def test_greedy_exploration__perturbs(self, method: str):
        exploration = E.GreedyExploration(strength=0.5, strength_decay_rate=0.75)
        exploration.perturbation(method)

    def test_epsilon_greedy_exploration__init(self):
        exploration = E.EpsilonGreedyExploration(
            epsilon=0.7, strength=0.5, epsilon_decay_rate=0.75
        )
        self.assertEqual(exploration.strength_decay_rate, 0.75)

    def test_epsilon_greedy_exploration__sometimes_explores(self):
        exploration = E.EpsilonGreedyExploration(
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
        exploration = E.EpsilonGreedyExploration(
            epsilon=0.7, strength=0.5, epsilon_decay_rate=0.75, strength_decay_rate=0.2
        )
        for _ in range(5):
            exploration.decay()
        np.testing.assert_allclose(exploration.epsilon, 0.7 * 0.75**5)
        np.testing.assert_allclose(exploration.strength, 0.5 * 0.2**5)


class TestParameters(unittest.TestCase):
    @parameterized.expand(map(lambda i: (i,), range(3)))
    def test_learnable_parameter_init__raises__with_unbroadcastable_args(self, i: int):
        args = [5, 0, 10]
        args[i] = np.random.rand(2, 2, 2)
        with self.assertRaises(ValueError):
            LearnableParameter("theta", 10, *args)

    def test_learnable_parameter_init_and_update__raise__with_value_outside_range(self):
        with self.assertRaises(ValueError):
            LearnableParameter("theta", 10, -5, 0, 10)
        par = LearnableParameter("theta", 10, 5, 0, 10)
        with self.assertRaises(ValueError):
            par._update_value(-5)

    @parameterized.expand(product([cs.SX, cs.MX], [False, True]))
    def test_learnable_parameters_dict__is_deepcopyable_and_pickleable(
        self, cstype: Union[cs.SX, cs.MX], copy: bool
    ):
        size = 5
        theta_sym = cstype.sym("theta", size)
        f = theta_sym[0]
        df = cs.evalf(cs.jacobian(f, theta_sym)[0])
        p1 = LearnableParameter[float]("theta", size, 1, -1, 2, sym={"v": theta_sym})
        pars = LearnableParametersDict((p1,))
        if copy:
            new_pars = pars.copy(deep=True)
        else:
            with pars.pickleable():
                new_pars: LearnableParametersDict = pickle.loads(pickle.dumps(pars))
        p2: LearnableParameter = new_pars["theta"]
        self.assertIsNot(p1, p2)
        self.assertEqual(p1.name, p2.name)
        self.assertEqual(p1.size, p2.size)
        np.testing.assert_array_equal(p1.value, p2.value)
        np.testing.assert_array_equal(p1.lb, p2.lb)
        np.testing.assert_array_equal(p1.ub, p2.ub)
        np.testing.assert_equal(df, cs.evalf(cs.jacobian(f, p1.sym["v"])[0]))
        if copy:
            self.assertIsNot(p1.sym, p2.sym)
            self.assertIsNot(p1.sym["v"], p2.sym["v"])
            np.testing.assert_equal(df, cs.evalf(cs.jacobian(f, p2.sym["v"])[0]))
        else:
            self.assertFalse(hasattr(p2, "sym"))

    @parameterized.expand([(False,), (True,)])
    def test_parameters_dict__init__initializes_properly(self, empty_init: bool):
        p1 = LearnableParameter("theta1", 10, 0.0)
        p2 = LearnableParameter("theta2", 11, 0.0)
        PARS = [p1, p2]
        if empty_init:
            pars = LearnableParametersDict()
            pars.update(iter(PARS))
        else:
            pars = LearnableParametersDict(iter(PARS))
        self.assertEqual(len(pars), 2)
        for p in PARS:
            self.assertIn(p.name, pars)
            self.assertIs(pars[p.name], p)

    @parameterized.expand([(False,), (True,)])
    def test_parameters_dict__setitem__raises_with_wrong_name(self, wrong_name: bool):
        p = LearnableParameter("t", 10, 0.0)
        pars = LearnableParametersDict()
        if wrong_name:
            with self.assertRaisesRegex(
                AssertionError, r"Key 't_hello_there' must match parameter name 't'."
            ):
                pars[f"{p.name}_hello_there"] = p
        else:
            pars[p.name] = p

    @parameterized.expand(
        map(
            lambda m: (m,),
            ["setitem", "update", "setdefault", "delitem", "pop", "popitem", "clear"],
        )
    )
    def test_parameters_dict__caches_get_cleared_properly(self, method: str):
        assert_array_equal = np.testing.assert_array_equal

        def check(pars, PARS):
            self.assertEqual(pars.size, sum(p.size for p in PARS))
            assert_array_equal(pars.lb, sum((p.lb.tolist() for p in PARS), []))
            assert_array_equal(pars.ub, sum((p.ub.tolist() for p in PARS), []))
            assert_array_equal(pars.value, sum((p.value.tolist() for p in PARS), []))
            if len(pars) > 0 and len(PARS) > 0:
                sym1 = cs.vertcat(*(p.sym for p in pars.values()))
                sym2 = cs.vertcat(*(p.sym for p in PARS))
                assert_array_equal(cs.evalf(cs.simplify(sym1 - sym2)), 0)

        p1 = LearnableParameter[float]("t1", 1, 1.0, -1.0, 2.0, cs.SX.sym("t1", 1))
        p2 = LearnableParameter[float]("t2", 2, 2.0, -2.0, 3.0, cs.SX.sym("t2", 2))
        PARS = [p1, p2]
        pars = LearnableParametersDict(PARS)
        check(pars, PARS)

        p3 = LearnableParameter[float]("t3", 3, 3.0, -3.0, 4.0, cs.SX.sym("t3", 3))
        if method == "setitem":
            pars[p3.name] = p3
            PARS = [p1, p2, p3]
        elif method == "update":
            pars.update([p3])
            PARS = [p1, p2, p3]
        elif method == "setdefault":
            pars.setdefault(p3)
            PARS = [p1, p2, p3]
        elif method == "delitem":
            del pars[p2.name]
            PARS = [p1]
        elif method == "pop":
            pars.pop(p1.name)
            PARS = [p2]
        elif method == "popitem":
            pars.popitem()
            PARS = [p1]
        elif method == "clear":
            pars.clear()
            PARS = []

        check(pars, PARS)


if __name__ == "__main__":
    unittest.main()
