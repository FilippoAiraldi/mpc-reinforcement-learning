import pickle
import unittest
from collections.abc import Iterator
from copy import deepcopy
from itertools import product
from random import shuffle
from typing import Union
from unittest.mock import Mock

import casadi as cs
import numpy as np
from csnlp.multistart import RandomStartPoints, StructuredStartPoints
from parameterized import parameterized

from mpcrl import (
    ExperienceReplay,
    LearnableParameter,
    LearnableParametersDict,
    MpcSolverError,
    MpcSolverWarning,
    UpdateError,
    UpdateStrategy,
    UpdateWarning,
    WarmStartStrategy,
)
from mpcrl import exploration as E
from mpcrl import schedulers as S
from mpcrl.core.errors import (
    raise_or_warn_on_mpc_failure,
    raise_or_warn_on_update_failure,
)


def do_test_str_and_repr(testcase: unittest.TestCase, obj: S.Scheduler) -> None:
    testcase.assertIn(obj.__class__.__name__, obj.__str__())
    testcase.assertIn(obj.__class__.__name__, obj.__repr__())


def do_test_similar_schedulers(
    testcase: unittest.TestCase, obj1: S.Scheduler, obj2: S.Scheduler
) -> None:
    dict1 = obj1.__dict__
    dict2 = obj2.__dict__
    for name, val1 in dict1.items():
        val2 = dict2[name]
        if hasattr(val1, "__iter__"):
            testcase.assertListEqual(
                [next(val1) for _ in range(10)], [next(val2) for _ in range(10)]
            )
        else:
            testcase.assertEqual(val1, val2)


class TestErrors(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_raise_or_warn_on_mpc_failure__raises_or_warns(self, raises: bool):
        msg = "This is a message"
        context = (
            self.assertRaisesRegex(MpcSolverError, msg)
            if raises
            else self.assertWarnsRegex(MpcSolverWarning, msg)
        )
        with context:
            raise_or_warn_on_mpc_failure(msg, raises)

    @parameterized.expand([(False,), (True,)])
    def test_raise_or_warn_on_update_failure__raises_or_warns(self, raises: bool):
        msg = "This is a message"
        context = (
            self.assertRaisesRegex(UpdateError, msg)
            if raises
            else self.assertWarnsRegex(UpdateWarning, msg)
        )
        with context:
            raise_or_warn_on_update_failure(msg, raises)


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
        with self.assertRaises(TypeError):
            ExperienceReplay[tuple[np.ndarray, float]](maxlen=None, sample_size=0.0)

    @parameterized.expand([(0,), (float(0),)])
    def test_sample__with_zero_samples__returns_no_samples(self, n: Union[int, float]):
        mem = ExperienceReplay[tuple[np.ndarray, float]](maxlen=100, sample_size=n)
        self.assertListEqual(list(mem.sample()), [])

    @parameterized.expand([(10,), (0.1,)])
    def test_sample__returns_right_sample_size(self, n: Union[int, float]):
        N = 100
        Nsample = 10
        mem = ExperienceReplay[int](maxlen=N, sample_size=n)
        mem.extend(range(N))
        sample = list(mem.sample())
        self.assertEqual(len(sample), Nsample)
        for item in sample:
            self.assertIn(item, mem)

    @parameterized.expand([(20, 10), (20, 0.5), (0.2, 10), (0.2, 0.5)])
    def test_sample__latest_n__includes_latest_n_items(
        self, n: Union[int, float], last_n: Union[int, float]
    ):
        N = 100
        Nsample = 20
        Nlast = 10
        mem = ExperienceReplay[int](maxlen=N, sample_size=n, include_latest=last_n)
        mem.extend(range(N))
        sample = list(mem.sample())
        self.assertEqual(len(sample), Nsample)
        for item in sample:
            self.assertIn(item, mem)
        for item in range(N - Nlast, N):
            self.assertIn(item, sample)

    def test_deepcopy(self):
        maxlen = np.random.randint(10, 100)
        sample_size, include_latest = np.random.uniform(size=2)
        mem = ExperienceReplay[int](
            maxlen=maxlen, sample_size=sample_size, include_latest=include_latest
        )
        mem_copy = deepcopy(mem)
        self.assertEqual(mem.maxlen, mem_copy.maxlen)
        self.assertEqual(mem.sample_size, mem_copy.sample_size)
        self.assertEqual(mem.include_latest, mem_copy.include_latest)
        self.assertIsInstance(mem_copy.np_random, np.random.Generator)


class TestSchedulers(unittest.TestCase):
    def test_no_scheduling__step__does_not_update_value(self):
        scheduler = S.NoScheduling(init_value=None)
        scheduler.step()
        self.assertIsNone(scheduler.value)
        do_test_str_and_repr(self, scheduler)

    def test_exponential_scheduler__step__updates_value_correctly(self):
        K = 20
        x0 = np.random.randn()
        factor = np.random.rand()
        x_expected = x0 * factor ** np.arange(K)
        scheduler = S.ExponentialScheduler(x0, factor)
        x_actual = []
        for _ in range(K):
            x_actual.append(scheduler.value)
            scheduler.step()
        np.testing.assert_allclose(x_expected, x_actual)
        do_test_str_and_repr(self, scheduler)

    def test_linear_scheduler__step__updates_value_correctly(self):
        K = 20
        x0 = np.random.rand() * 10
        xf = np.random.rand() * 10
        x_expected = np.linspace(x0, xf, K + 1)
        scheduler = S.LinearScheduler(x0, xf, K)
        x_actual = []
        for _ in range(K):
            x_actual.append(scheduler.value)
            scheduler.step()
        x_actual.append(scheduler.value)
        with self.assertRaises(StopIteration):
            scheduler.step()
        np.testing.assert_allclose(x_expected, x_actual)
        do_test_str_and_repr(self, scheduler)

    def test_loglinear_scheduler__step__updates_value_correctly(self):
        K = 20
        x0 = np.abs(2 * np.random.randn())
        xf = np.abs(2 * np.random.randn())
        x_expected = np.geomspace(x0, xf, K + 1)
        scheduler = S.LogLinearScheduler(x0, xf, K)
        x_actual = []
        for _ in range(K):
            x_actual.append(scheduler.value)
            scheduler.step()
        x_actual.append(scheduler.value)
        with self.assertRaises(StopIteration):
            scheduler.step()
        np.testing.assert_allclose(x_expected, x_actual)
        do_test_str_and_repr(self, scheduler)

    def test_deepcopy(self):
        K = np.random.randint(10, 20)
        base = np.random.rand() + 1 * 10
        x0 = np.random.randn() + 3
        xf = np.random.randn()
        scheduler = S.LogLinearScheduler(base**x0, base**xf, K)
        scheduler_copy = deepcopy(scheduler)
        do_test_similar_schedulers(self, scheduler, scheduler_copy)

    def test_chain(self):
        # create a list of random schedulers
        rd, ri = np.random.rand, np.random.randint
        mk1 = lambda: (S.NoScheduling(rd()), ri(10, 100))
        mk2 = lambda: (S.ExponentialScheduler(*rd(2)), ri(10, 100))
        mk3 = lambda: S.LinearScheduler(*rd(2), ri(10, 100))
        mk4 = lambda: S.LogLinearScheduler(*rd(2), ri(10, 100))
        schedulers = []
        for mk in (mk1, mk2, mk3, mk4):
            for _ in range(ri(1, 5)):
                schedulers.append(mk())
        shuffle(schedulers)

        # extract the expected values
        values_expected = []
        schedulers_ = deepcopy(schedulers)
        for scheduler in schedulers_:
            if isinstance(scheduler, tuple):
                scheduler, K = scheduler
                while True:
                    values_expected.append(scheduler.value)
                    scheduler.step()
                    K -= 1
                    if K == 0:
                        break
            else:
                while True:
                    values_expected.append(scheduler.value)
                    try:
                        scheduler.step()
                    except StopIteration:
                        break

        # create the chain and extract the actual values
        scheduler = S.Chain(schedulers)
        values_actual = []
        while True:
            values_actual.append(scheduler.value)
            try:
                scheduler.step()
            except StopIteration:
                break

        # test
        steps_expected = sum(
            sc[1] if isinstance(sc, tuple) else sc.total_steps + 1 for sc in schedulers
        )
        self.assertEqual(len(values_actual), steps_expected)
        self.assertEqual(len(values_expected), len(values_actual))
        np.testing.assert_array_equal(values_expected, values_actual)


class TestExploration(unittest.TestCase):
    def test_no_exploration__has_no_mode_nor_hook(self):
        exploration = E.NoExploration()
        self.assertFalse(hasattr(exploration, "_mode"), "should not have `mode`")
        self.assertFalse(hasattr(exploration, "_hook"), "should not have `_hook`")
        self.assertIsNone(exploration.hook)
        self.assertIsNone(exploration.mode)

    def test_no_exploration__never_explores(self):
        exploration = E.NoExploration()
        self.assertFalse(exploration.can_explore())
        with self.assertRaisesRegex(
            NotImplementedError, "Perturbation not implemented in NoExploration"
        ):
            exploration.perturbation()
        exploration.step()  # does nothing
        do_test_str_and_repr(self, exploration)

    def test_greedy_exploration__instantiates_np_random(self):
        exploration = E.GreedyExploration(strength=0.5)
        self.assertIsInstance(exploration.np_random, np.random.Generator)

    def test_greedy_exploration__always_explores(self):
        exploration = E.GreedyExploration(strength=0.5)
        self.assertTrue(exploration.can_explore())
        do_test_str_and_repr(self, exploration)

    @parameterized.expand(((False,), (True,)))
    def test_greedy_exploration__hook(self, strength):
        strength_scheduler = S.LinearScheduler(0, 1, 10) if strength else 0
        exploration = E.GreedyExploration(strength_scheduler)
        hook = exploration.hook
        self.assertEqual(hook is not None, strength)

    @parameterized.expand([("uniform",), ("normal",), ("standard_normal",)])
    def test_greedy_exploration__perturbs(self, method: str):
        exploration = E.GreedyExploration(strength=0.5)
        exploration.perturbation(method)

    def test_epsilon_greedy_exploration__never_explores__with_zero_epsilon(self):
        epsilon, epsilon_decay_rate = 0.0, 0.75
        strength, strength_decay_rate = 0.5, 0.75
        epsilon_scheduler = S.ExponentialScheduler(epsilon, epsilon_decay_rate)
        strength_scheduler = S.ExponentialScheduler(strength, strength_decay_rate)
        exploration = E.EpsilonGreedyExploration(
            epsilon=epsilon_scheduler, strength=strength_scheduler, seed=42
        )
        self.assertFalse(any(exploration.can_explore() for _ in range(100)))

    def test_epsilon_greedy_exploration__sometimes_explores(self):
        epsilon, epsilon_decay_rate = 0.7, 0.75
        strength, strength_decay_rate = 0.5, 0.75
        epsilon_scheduler = S.ExponentialScheduler(epsilon, epsilon_decay_rate)
        strength_scheduler = S.ExponentialScheduler(strength, strength_decay_rate)
        exploration = E.EpsilonGreedyExploration(
            epsilon=epsilon_scheduler, strength=strength_scheduler, seed=42
        )
        self.assertFalse(exploration.can_explore())
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
        class MockScheduler(S.NoScheduling): ...

        epsilon_scheduler = MockScheduler(None)
        strength_scheduler = MockScheduler(None)
        epsilon_scheduler.step = Mock()
        strength_scheduler.step = Mock()
        exploration = E.EpsilonGreedyExploration(
            epsilon=epsilon_scheduler, strength=strength_scheduler, seed=42
        )

        exploration.step()

        epsilon_scheduler.step.assert_called_once()
        strength_scheduler.step.assert_called_once()

    @parameterized.expand(product([False, True], [False, True]))
    def test_epsilon_greedy_exploration__hook(self, epsilon, strength):
        epsilon_scheduler = S.LinearScheduler(0, 1, 10) if epsilon else 0
        strength_scheduler = S.LinearScheduler(0, 1, 10) if strength else 0
        exploration = E.EpsilonGreedyExploration(epsilon_scheduler, strength_scheduler)
        hook = exploration.hook
        self.assertEqual(hook is not None, epsilon or strength)

    def test_ornsteinuhlenbeck_exploration__always_explores(self):
        exploration = E.OrnsteinUhlenbeckExploration(0, 0.5)
        self.assertTrue(exploration.can_explore())

    def test_ornsteinuhlenbeck_exploration__decays_mean_and_sigma(self):
        class MockScheduler(S.NoScheduling): ...

        mean_scheduler = MockScheduler(None)
        sigma_scheduler = MockScheduler(None)
        mean_scheduler.step = Mock()
        sigma_scheduler.step = Mock()
        exploration = E.OrnsteinUhlenbeckExploration(mean_scheduler, sigma_scheduler)

        exploration.step()

        mean_scheduler.step.assert_called_once()
        sigma_scheduler.step.assert_called_once()

    @parameterized.expand(product([False, True], [False, True]))
    def test_ornsteinuhlenbeck_exploration__hook(self, mean, sigma):
        mean_scheduler = S.LinearScheduler(0, 1, 10) if mean else 0
        sigma_scheduler = S.LinearScheduler(0, 1, 10) if sigma else 0
        exploration = E.OrnsteinUhlenbeckExploration(mean_scheduler, sigma_scheduler)
        hook = exploration.hook
        self.assertEqual(hook is not None, mean or sigma)

    def test_stepwise_exploration__has_same_hook_and_mode_as_base_exploration(self):
        hook, mode = Mock(), Mock()
        base_exploration = Mock()
        base_exploration.hook = hook
        base_exploration.mode = mode
        exploration = E.StepWiseExploration(base_exploration, 5, 10)
        self.assertIs(exploration.hook, hook)
        self.assertIs(exploration.mode, mode)

    @parameterized.expand([(True,), (False,)])
    def test_stepwise_exploration__turns_base_exploration_into_steps(
        self, stepwise_step: bool
    ):
        base_exploration = E.EpsilonGreedyExploration(0.5, 0.5, seed=0)
        base_exploration.step = Mock()
        step_size = 5
        cycles = 10
        exploration = E.StepWiseExploration(base_exploration, step_size, stepwise_step)
        can_explores, perturbations = [], []
        for _ in range(cycles):
            can_explores.append([])
            perturbations.append([])
            for _ in range(step_size):
                can_explores[-1].append(exploration.can_explore())
                perturbations[-1].append(exploration.perturbation("uniform"))
                exploration.step()

        for can_explores_cycle, perturbations_cycle in zip(can_explores, perturbations):
            self.assertTrue(np.unique(can_explores_cycle).size == 1)
            self.assertTrue(np.unique(perturbations_cycle).size == 1)
        if stepwise_step:
            self.assertTrue(len(base_exploration.step.mock_calls) == cycles)
        else:
            self.assertTrue(len(base_exploration.step.mock_calls) == cycles * step_size)

    def test_deepcopy(self):
        epsilon, epsilon_decay_rate = 0.0, 0.75
        strength, strength_decay_rate = 0.5, 0.75
        epsilon_scheduler = S.ExponentialScheduler(epsilon, epsilon_decay_rate)
        strength_scheduler = S.ExponentialScheduler(strength, strength_decay_rate)
        exploration = E.EpsilonGreedyExploration(
            epsilon=epsilon_scheduler, strength=strength_scheduler, seed=42
        )
        exploration_copy = deepcopy(exploration)
        self.assertEqual(exploration._hook, exploration_copy._hook)
        do_test_similar_schedulers(
            self, exploration.epsilon_scheduler, exploration_copy.epsilon_scheduler
        )
        do_test_similar_schedulers(
            self, exploration.strength_scheduler, exploration_copy.strength_scheduler
        )


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
        array_equal = np.testing.assert_array_equal
        cat = np.concatenate

        def check(pars, PARS):
            self.assertEqual(pars.size, sum(p.size for p in PARS))
            if len(pars) > 0 and len(PARS) > 0:
                array_equal(pars.lb, cat([p.lb.flatten("F") for p in PARS]))
                array_equal(pars.ub, cat([p.ub.flatten("F") for p in PARS]))
                array_equal(pars.value, cat([p.value.flatten("F") for p in PARS]))
                sym1 = cs.veccat(*(p.sym for p in pars.values()))
                sym2 = cs.veccat(*(p.sym for p in PARS))
                array_equal(cs.evalf(cs.simplify(sym1 - sym2)), 0)
            else:
                self.assertTupleEqual(pars.lb.shape, (0,))
                self.assertTupleEqual(pars.ub.shape, (0,))
                self.assertTupleEqual(pars.value.shape, (0,))

        p1 = LearnableParameter[float](
            "t1", (4, 6), 1.0, -1.0, 2.0, cs.SX.sym("t1", 4, 6)
        )
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

    def test_parameters_dict__stringify(self):
        p1 = LearnableParameter[None]("p1", 1, 1)
        p2 = LearnableParameter[None]("p2", 1, 2.0)
        p3 = LearnableParameter[None]("p3", 100, np.random.randint(0, 100, size=100))
        p4 = LearnableParameter[None]("p4", 100, np.random.rand(100))
        pars = LearnableParametersDict((p1, p2, p3, p4))
        S = pars.stringify()
        for p in [p1, p2, p3, p4]:
            self.assertIn(p.name, S)

    @parameterized.expand(product([cs.SX, cs.MX], [False, True]))
    def test_deepcopy(self, cstype: Union[cs.SX, cs.MX], copy: bool):
        shape = (5, 2)
        theta_sym = cstype.sym("theta", shape)
        f = theta_sym[0]
        df = cs.evalf(cs.jacobian(f, theta_sym)[0])
        p1 = LearnableParameter[float]("theta", shape, 1, -1, 2, sym={"v": theta_sym})
        pars = LearnableParametersDict((p1,))
        if copy:
            new_pars = pars.copy(deep=True)
        else:
            new_pars: LearnableParametersDict = pickle.loads(pickle.dumps(pars))
        p2: LearnableParameter = new_pars["theta"]
        self.assertIsNot(p1, p2)
        self.assertEqual(p1.name, p2.name)
        self.assertEqual(p1.shape, p2.shape)
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


class TestUpdateStrategy(unittest.TestCase):
    def test__is_iterable(self):
        strategy = UpdateStrategy(5)
        self.assertIsInstance(next(strategy), bool)
        self.assertIsInstance(iter(strategy), Iterator)

    def test_can_update__has_right_frequency(self):
        N = 50
        freq = 3
        strategy = UpdateStrategy(freq)
        flags = np.asarray([strategy.can_update() for _ in range(N)])
        mask = np.asarray(([False] * (freq - 1) + [True]) * (N // freq + 1))[:N]
        self.assertEqual(flags.sum(), N // freq)
        self.assertTrue(flags[mask].all())
        self.assertTrue((~flags[~mask]).all())

    def test_can_update__with_skip_first__skips_first_updates_correctly(self):
        N = 50
        freq = 3
        skip = 2
        strategy = UpdateStrategy(freq, skip_first=skip)
        flags = np.asarray([strategy.can_update() for _ in range(N)])
        mask = np.asarray(([False] * (freq - 1) + [True]) * (N // freq + 1))[:N]
        self.assertEqual(flags.sum(), N // freq - skip)
        self.assertTrue((~flags[mask][:skip]).all())
        self.assertTrue(flags[mask][skip:].all())
        self.assertTrue((~flags[~mask]).all())

    def test_deepcopy(self):
        strategy = UpdateStrategy(5, "on_episode_end", 3)
        strategy_copy = deepcopy(strategy)
        self.assertEqual(strategy.frequency, strategy_copy.frequency)
        self.assertEqual(strategy.hook, strategy_copy.hook)
        self.assertListEqual(
            [next(strategy._update_cycle) for _ in range(50)],
            [next(strategy_copy._update_cycle) for _ in range(50)],
        )


class TestWarmStartStrategy(unittest.TestCase):
    @parameterized.expand(product((False, True), (False, True)))
    def test_n_points(self, struct_points: bool, rand_points: bool):
        ns = np.random.randint(3, 100) if struct_points else 0
        nr = np.random.randint(3, 100) if rand_points else 0
        wss = WarmStartStrategy(
            structured_points=StructuredStartPoints({}, ns) if struct_points else None,
            random_points=RandomStartPoints({}, nr) if rand_points else None,
        )
        self.assertEqual(wss.n_points, ns + nr)

    def test_reset(self):
        wss = WarmStartStrategy()
        random_points = Mock()
        wss.random_points = random_points
        wss.reset(42)
        self.assertIsInstance(random_points.np_random, np.random.Generator)

    def test_generate(self):
        struct_point = {"a": object(), "b": object()}
        rand_point = {"a": object(), "b": object()}
        struct_points = Mock()
        rand_points = Mock()
        struct_points.__iter__ = lambda _: iter([struct_point])
        rand_points.__iter__ = lambda _: iter([rand_point])
        prev_sol = {"a": object(), "b": object(), "c": object()}
        wss = WarmStartStrategy(
            structured_points=struct_points,
            random_points=rand_points,
            update_biases_for_random_points=True,
        )

        out = list(wss.generate(previous_sol=prev_sol))

        rand_points.biases.update.assert_called_once_with(prev_sol)
        self.assertDictEqual(out[0], {**struct_point, "c": prev_sol["c"]})
        self.assertDictEqual(out[1], {**rand_point, "c": prev_sol["c"]})


if __name__ == "__main__":
    unittest.main()
