import random
import unittest
from math import cos, pi, sqrt
from random import Random
from unittest.mock import Mock, patch

from qutai import Tensor
from qutai.core.context import is_recording, is_training, set_recording, set_training
from qutai.generation.diffusion import Diffusion, FlowMatching, normal


class ConstantModel:
    def __init__(self, value):
        self.parameter = Tensor(value, requires_grad=True)
        self.parameter.grad = Tensor(9)
        self.calls = []

    def __call__(self, state, times, *, train=True):
        self.calls.append((times[:], train, is_recording(), is_training()))
        return state * 0 + self.parameter


class NoiseOracle:
    def __init__(self, method, clean):
        self.method, self.clean = method, clean

    def __call__(self, state, times, *, train=True):
        step = round(times[0] * self.method.steps)
        alpha_bar = self.method.alpha_bars[step]
        return (state - sqrt(alpha_bar) * self.clean) / sqrt(1 - alpha_bar)


class GenerationMethodTests(unittest.TestCase):
    def test_normal_uses_supplied_random_and_preserves_shape(self):
        reference = Random(42)
        expected = [reference.gauss(0, 1) for _ in range(4)]
        noise = normal((2, 1, 1, 2), Random(42))
        self.assertEqual(noise.shape, (2, 1, 1, 2))
        self.assertEqual(noise.values, expected)
        self.assertFalse(noise.requires_grad)

    def test_two_step_cosine_schedule_and_ddpm_posterior_match_hand_calculation(self):
        method = Diffusion(2)
        half = cos((0.5 + 0.008) / 1.008 * pi / 2) ** 2 / cos(0.008 / 1.008 * pi / 2) ** 2
        self.assertAlmostEqual(method.betas[0], 1 - half)
        self.assertEqual(method.betas[1], 0.999)
        self.assertEqual(method.alpha_bars[0], 1)
        self.assertAlmostEqual(method.alpha_bars[1], half)
        self.assertAlmostEqual(method.alpha_bars[2], half * 0.001)
        self.assertGreater(method.alpha_bars[2], 0)

        random = Random(7)
        initial = [random.gauss(0, 1) for _ in range(2)]
        added = [random.gauss(0, 1) for _ in range(2)]
        variance = 0.999 * (1 - half) / (1 - half * 0.001)
        middle = [value / sqrt(0.001) + sqrt(variance) * noise for value, noise in zip(initial, added, strict=True)]
        final = [value / sqrt(half) for value in middle]
        model = ConstantModel(0)
        with patch('qutai.generation.diffusion.normal', wraps=normal) as gaussian:
            states = list(method.states(model, (1, 1, 1, 2), seed=7))
            self.assertEqual(gaussian.call_count, 2)
        self.assertEqual(len(states), 3)
        for state, expected in zip(states, (initial, middle, final), strict=True):
            for value, reference in zip(state.values, expected, strict=True):
                self.assertAlmostEqual(value, reference, places=11)
        self.assertEqual([call[0] for call in model.calls], [[1.0], [0.5]])
        self.assertTrue(any(abs(value) > 1 for value in states[-1].values))

    def test_diffusion_training_times_and_clipped_schedule_coefficients(self):
        method = Diffusion(2)
        images = Tensor([2, -3], shape=(2, 1, 1, 1))
        random = Mock()
        random.randint.side_effect = [1, 2]
        random.gauss.return_value = 0.25
        mixed, times, target = method.prepare(images, random)
        self.assertEqual(times, [0.5, 1.0])
        self.assertEqual(mixed.shape, images.shape)
        self.assertEqual(target.shape, images.shape)
        self.assertEqual(target.values, [0.25, 0.25])
        self.assertFalse(target.requires_grad)
        for value, image, alpha_bar in zip(mixed.values, images.values, method.alpha_bars[1:], strict=True):
            self.assertAlmostEqual(value, sqrt(alpha_bar) * image + sqrt(1 - alpha_bar) * 0.25)
        self.assertEqual(images.values, [2, -3])

    def test_perfect_noise_prediction_recovers_unclipped_clean_image(self):
        clean = Tensor([-0.4, 2.5], shape=(1, 1, 1, 2))
        for steps in (1, 4):
            with self.subTest(steps=steps):
                method = Diffusion(steps)
                states = list(method.states(NoiseOracle(method, clean), clean.shape, seed=42))
                self.assertEqual(len(states), steps + 1)
                for value, expected in zip(states[-1].values, clean.values, strict=True):
                    self.assertAlmostEqual(value, expected, places=11)

    def test_flow_training_path_endpoints_and_velocity(self):
        images = Tensor([2, -3], shape=(2, 1, 1, 1))
        random = Mock()
        random.random.side_effect = [0.0, 1.0]
        random.gauss.return_value = 0.25
        mixed, times, target = FlowMatching(2).prepare(images, random)
        self.assertEqual(times, [0, 1])
        self.assertEqual(mixed.shape, images.shape)
        self.assertEqual(target.shape, images.shape)
        self.assertEqual(mixed.values, [0.25, -3])
        self.assertEqual(target.values, [1.75, -3.25])
        self.assertEqual(images.values, [2, -3])

    def test_constant_velocity_follows_exact_euler_path(self):
        method, model = FlowMatching(4), ConstantModel(2.5)
        shape = (1, 1, 1, 2)
        states = list(method.states(model, shape, seed=7))
        initial = normal(shape, Random(7)).values
        self.assertEqual(len(states), 5)
        for step, state in enumerate(states):
            for value, start in zip(state.values, initial, strict=True):
                self.assertAlmostEqual(value, start + 2.5 * step / 4)
        self.assertEqual([call[0] for call in model.calls], [[0], [0.25], [0.5], [0.75]])
        self.assertGreater(max(states[-1].values), 1)

    def test_every_yield_restores_context_and_preserves_parameter_gradients(self):
        for method in (Diffusion(2), FlowMatching(2)):
            for enabled in (True, False):
                with self.subTest(method=type(method).__name__, enabled=enabled):
                    model = ConstantModel(0.3)
                    gradient = model.parameter.grad
                    with set_recording(enabled), set_training(enabled):
                        outside = is_recording(), is_training()
                        for state in method.states(model, (2, 1, 1, 2), seed=7):
                            self.assertEqual((is_recording(), is_training()), outside)
                            self.assertFalse(state.requires_grad)
                            self.assertEqual(state.parents, ())
                            self.assertIsNone(state.backward_rule)
                            self.assertIs(model.parameter.grad, gradient)
                    self.assertTrue(all(call[1:] == (False, False, False) for call in model.calls))
                    self.assertEqual(model.parameter.item(), 0.3)

    def test_seed_reproducibility_without_global_random_changes(self):
        images = Tensor([0.1, 0.2, 0.3, 0.4], shape=(2, 1, 1, 2))
        before = random.getstate()
        for method in (Diffusion(2), FlowMatching(2)):
            first = method.prepare(images, Random(42))
            second = method.prepare(images, Random(42))
            other = method.prepare(images, Random(7))
            self.assertEqual((first[0].values, first[1], first[2].values), (second[0].values, second[1], second[2].values))
            self.assertNotEqual(first[0].values, other[0].values)
            initial = [state.values for state in method.states(ConstantModel(0), images.shape, seed=42)]
            repeated = [state.values for state in method.states(ConstantModel(0), images.shape, seed=42)]
            self.assertEqual(initial, repeated)
        self.assertEqual(random.getstate(), before)

    def test_config_and_positive_step_requirement(self):
        self.assertEqual(Diffusion(2).config(), {'type': 'Diffusion', 'steps': 2, 'schedule': 'cosine'})
        self.assertEqual(FlowMatching(2).config(), {'type': 'FlowMatching', 'steps': 2, 'solver': 'euler'})
        for method in (Diffusion, FlowMatching):
            for steps in (0, -1, True, 1.5):
                with self.subTest(method=method.__name__, steps=steps):
                    with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                        method(steps)


if __name__ == '__main__':
    unittest.main()
