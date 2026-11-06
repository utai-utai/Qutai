"""全局梯度裁剪的数值结果、共享参数与梯度生命周期。"""

import math
import unittest

from qutai import Tensor, nn


class GradientClippingTests(unittest.TestCase):
    def test_global_norm_scales_all_gradients_by_the_same_factor(self):
        left, right = Tensor([1, 2], requires_grad=True), Tensor(3, requires_grad=True)
        left.grad, right.grad = Tensor([3, 4]), Tensor(12)
        gradients = left.grad, right.grad
        self.assertEqual(nn.clip_grad_norm([left, right], max_norm=6.5), 13)
        self.assertEqual(left.grad.tolist(), [1.5, 2])
        self.assertEqual(right.grad.item(), 6)
        self.assertEqual((left.tolist(), right.item()), ([1, 2], 3))
        self.assertIs(left.grad, gradients[0])
        self.assertIs(right.grad, gradients[1])
        self.assertFalse(left.grad.requires_grad)
        self.assertEqual(left.grad.parents, ())

    def test_small_and_zero_gradients_are_unchanged(self):
        parameter = Tensor([1, 2])
        parameter.grad = Tensor([0.3, 0.4])
        self.assertAlmostEqual(nn.clip_grad_norm([parameter]), 0.5)
        self.assertEqual(parameter.grad.tolist(), [0.3, 0.4])
        parameter.grad = Tensor([0, 0])
        self.assertEqual(nn.clip_grad_norm([parameter], max_norm=0), 0)
        self.assertEqual(parameter.grad.tolist(), [0, 0])

    def test_zero_limit_removes_existing_gradient(self):
        parameter = Tensor(2)
        parameter.grad = Tensor(-3)
        self.assertEqual(nn.clip_grad_norm([parameter], max_norm=0), 3)
        self.assertEqual(parameter.grad.item(), 0)

    def test_shared_parameters_are_counted_once_and_missing_gradients_skipped(self):
        parameter, unused = Tensor([1, 2]), Tensor(8)
        parameter.grad = Tensor([3, 4])
        self.assertEqual(nn.clip_grad_norm(iter([parameter, unused, parameter]), max_norm=2.5), 5)
        self.assertEqual(parameter.grad.tolist(), [1.5, 2])
        self.assertIsNone(unused.grad)
        self.assertEqual(nn.clip_grad_norm([unused]), 0)
        self.assertEqual(nn.clip_grad_norm([]), 0)

    def test_model_input_uses_only_declared_parameters(self):
        layer = nn.Linear(1, 1, seed=42)
        model = nn.Model([layer, layer])
        layer.weight.grad, layer.bias.grad = Tensor([[3]]), Tensor([4])
        model.unrelated = Tensor(10)
        model.unrelated.grad = Tensor(100)
        self.assertEqual(nn.clip_grad_norm(model, max_norm=2.5), 5)
        self.assertEqual(layer.weight.grad.item(), 1.5)
        self.assertEqual(layer.bias.grad.item(), 2)
        self.assertEqual(model.unrelated.grad.item(), 100)

    def test_large_finite_gradients_do_not_overflow_when_finding_norm(self):
        parameter = Tensor([0, 0])
        parameter.grad = Tensor([3e200, 4e200])
        self.assertAlmostEqual(nn.clip_grad_norm([parameter], max_norm=1) / 1e200, 5)
        self.assertAlmostEqual(parameter.grad.values[0], 0.6)
        self.assertAlmostEqual(parameter.grad.values[1], 0.8)

    def test_invalid_limit_or_nonfinite_gradient_fails_before_mutating(self):
        parameter = Tensor(2)
        parameter.grad = Tensor(3)
        for limit in (-1, math.nan, math.inf):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                nn.clip_grad_norm([parameter], max_norm=limit)
            self.assertEqual(parameter.grad.item(), 3)
        bad = Tensor(0)
        for value in (math.nan, math.inf, -math.inf):
            bad.grad = Tensor(value)
            with self.subTest(gradient=value), self.assertRaises(ValueError):
                nn.clip_grad_norm([parameter, bad])
            self.assertEqual(parameter.grad.item(), 3)


if __name__ == "__main__":
    unittest.main()
