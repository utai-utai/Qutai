"""AdamW 独立手算、缺失梯度与参数共享。"""

import math
import unittest

from qutai import Tensor, nn
from qutai.nn.optimizers import AdamW


class AdamWTests(unittest.TestCase):
    def test_bias_correction_and_decoupled_decay_by_hand(self):
        parameter = Tensor([2, -3], requires_grad=True)
        optimizer = AdamW([parameter], learning_rate=0.1, betas=(0.5, 0.75), eps=0, weight_decay=0.1)
        gradient = Tensor([2, -4])
        parameter.grad = gradient
        optimizer.step()
        self.assertAlmostEqual(parameter.values[0], 1.88)
        self.assertAlmostEqual(parameter.values[1], -2.87)
        self.assertEqual(optimizer.averages[parameter], [1, -2])
        self.assertEqual(optimizer.square_averages[parameter], [1, 4])
        self.assertIs(parameter.grad, gradient)
        self.assertEqual(gradient.tolist(), [2, -4])

        parameter.grad = Tensor([-1, 2])
        optimizer.step()
        self.assertEqual(optimizer.averages[parameter], [0, 0])
        self.assertEqual(optimizer.square_averages[parameter], [1, 4])
        self.assertEqual(optimizer.steps[parameter], 2)
        self.assertAlmostEqual(parameter.values[0], 1.8612)
        self.assertAlmostEqual(parameter.values[1], -2.8413)

    def test_nonzero_second_step_uses_both_bias_corrections_and_epsilon(self):
        parameter = Tensor(2, requires_grad=True)
        optimizer = AdamW([parameter], learning_rate=0.2, betas=(0.5, 0.75), eps=0.1, weight_decay=0)
        parameter.grad = Tensor(2)
        optimizer.step()
        first = 2 - 0.2 * 2 / 2.1
        self.assertAlmostEqual(parameter.item(), first)
        parameter.grad = Tensor(4)
        optimizer.step()
        # m2=.5*1+.5*4=2.5, v2=.75*1+.25*16=4.75。
        expected = first - 0.2 * (2.5 / 0.75) / (math.sqrt(4.75 / 0.4375) + 0.1)
        self.assertAlmostEqual(parameter.item(), expected)

    def test_missing_gradient_does_not_decay_or_advance_parameter(self):
        left, right = Tensor(1, requires_grad=True), Tensor(2, requires_grad=True)
        optimizer = AdamW([left, right], learning_rate=0.1, weight_decay=0.2)
        left.grad = Tensor(1)
        optimizer.step()
        self.assertEqual(right.item(), 2)
        self.assertNotIn(right, optimizer.steps)
        old_left = left.item()
        left.zero_grad()
        right.grad = Tensor(2)
        optimizer.step()
        self.assertEqual(left.item(), old_left)
        self.assertEqual(optimizer.steps[left], 1)
        self.assertEqual(optimizer.steps[right], 1)
        self.assertAlmostEqual(right.item(), 1.86 + 0.1 * 1e-8 / (2 + 1e-8))
        left.grad = Tensor(3)
        right.zero_grad()
        optimizer.step()
        self.assertEqual(optimizer.steps[left], 2)
        self.assertEqual(optimizer.steps[right], 1)

    def test_zero_gradient_still_decays_and_has_state(self):
        parameter = Tensor(2, requires_grad=True)
        parameter.grad = Tensor(0)
        optimizer = AdamW([parameter], learning_rate=0.1, weight_decay=0.5)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.9)
        self.assertEqual(optimizer.steps[parameter], 1)
        self.assertEqual(optimizer.averages[parameter], [0])
        self.assertEqual(optimizer.square_averages[parameter], [0])

    def test_shared_generator_parameter_is_updated_once(self):
        parameter = Tensor([[1, 2]], requires_grad=True)
        parameter.grad = Tensor([[1, -1]])
        optimizer = AdamW((item for item in (parameter, parameter)), learning_rate=0.1, weight_decay=0)
        optimizer.step()
        self.assertEqual(len(optimizer.parameters), 1)
        self.assertIs(optimizer.parameters[0], parameter)
        self.assertEqual(optimizer.steps[parameter], 1)
        self.assertAlmostEqual(parameter.values[0], 0.9 + 0.1 * 1e-8 / (1 + 1e-8))
        self.assertAlmostEqual(parameter.values[1], 2.1 - 0.1 * 1e-8 / (1 + 1e-8))
        self.assertEqual(parameter.shape, (1, 2))
        self.assertTrue(parameter.requires_grad)
        self.assertEqual(parameter.parents, ())
        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)

    def test_accepts_layer_and_model(self):
        layer = nn.Linear(1, 1, seed=42)
        model = nn.Model([layer, layer])
        for source in (layer, model):
            optimizer = AdamW(source)
            self.assertEqual(optimizer.parameters, layer.parameters())
        optimizer.zero_grad()
        loss = model([[1]]).square().sum()
        loss.backward()
        before = [parameter.values[:] for parameter in model.parameters()]
        optimizer.step()
        self.assertNotEqual([parameter.values for parameter in model.parameters()], before)
        self.assertTrue(all(optimizer.steps[parameter] == 1 for parameter in model.parameters()))

    def test_invalid_hyperparameters(self):
        for options in ({"learning_rate": -1}, {"weight_decay": -1}, {"eps": -1},
                        {"betas": (-0.1, 0.9)}, {"betas": (0.9, 1)}, {"betas": (0.9,)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AdamW([], **options)


if __name__ == "__main__":
    unittest.main()
