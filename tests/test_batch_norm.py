"""批归一化的逐通道统计、训练/预测梯度与运行状态。"""

import math
import unittest

from qutai import Tensor
from qutai.core.context import set_recording
from qutai.core.tensor import batch_norm2d


class BatchNormTests(unittest.TestCase):
    def check_gradients(self, train: bool) -> None:
        data = [0.2, -0.4, 1.3, 0.8, 0.7, 0.1, -0.2, 1.8]
        shapes = ((2, 2, 1, 2), (2,), (2,))
        values = (data, [1.2, -0.7], [0.3, -0.2])
        parents = [Tensor(items, shape=shape, requires_grad=True) for items, shape in zip(values, shapes)]
        seed = [0.2, -0.7, 1.3, 0.6, -0.3, 0.8, 0.4, -0.5]
        result = batch_norm2d(*parents, [0.1, -0.5], [0.8, 1.4], train=train, eps=0.2)
        result.backward(Tensor(seed, shape=result.shape))
        for parent_index, parent in enumerate(parents):
            for position, analytic in enumerate(parent.grad.values):
                high = [Tensor(items, shape=shape) for items, shape in zip(values, shapes)]
                low = [Tensor(items, shape=shape) for items, shape in zip(values, shapes)]
                high[parent_index].values[position] += 1e-6
                low[parent_index].values[position] -= 1e-6
                upper = batch_norm2d(*high, [0.1, -0.5], [0.8, 1.4], train=train, eps=0.2)
                lower = batch_norm2d(*low, [0.1, -0.5], [0.8, 1.4], train=train, eps=0.2)
                numeric = math.fsum((a - b) * incoming for a, b, incoming in zip(upper.values, lower.values, seed)) / 2e-6
                self.assertAlmostEqual(analytic, numeric, places=7)

    def test_training_gradients_by_central_difference(self):
        self.check_gradients(train=True)

    def test_evaluation_gradients_by_central_difference(self):
        self.check_gradients(train=False)

    def test_training_statistics_and_explicit_output_gradient(self):
        inputs = Tensor([1, 3], shape=(2, 1, 1, 1), requires_grad=True)
        weight, bias = Tensor([2], requires_grad=True), Tensor([1], requires_grad=True)
        running_mean, running_var = [10.0], [4.0]
        original_mean, original_var = running_mean, running_var
        result = batch_norm2d(inputs, weight, bias, running_mean, running_var, momentum=0.25, eps=3)
        self.assertEqual(result.values, [0, 2])
        self.assertIs(running_mean, original_mean)
        self.assertIs(running_var, original_var)
        self.assertEqual(running_mean, [8])
        self.assertEqual(running_var, [3.5])
        result.backward(Tensor([3, -1], shape=result.shape))
        self.assertEqual(inputs.grad.values, [1.5, -1.5])
        self.assertEqual(weight.grad.values, [-2])
        self.assertEqual(bias.grad.values, [2])

    def test_channels_and_batches_do_not_mix_statistics(self):
        inputs = Tensor([1, 3, 10, 14, 5, 7, 18, 22], shape=(2, 2, 1, 2))
        running_mean, running_var = [0.0, 10.0], [1.0, 4.0]
        result = batch_norm2d(inputs, [2, -0.5], [1, 3], running_mean, running_var, momentum=0.25, eps=4)
        self.assertEqual(running_mean, [1, 11.5])
        self.assertAlmostEqual(running_var[0], 29 / 12)
        self.assertAlmostEqual(running_var[1], 29 / 3)
        self.assertEqual(result.values[0], -1)
        self.assertEqual(result.values[5], 3)
        self.assertAlmostEqual(result.values[2], 3 + 3 / math.sqrt(24))
        self.assertAlmostEqual(result.values[7], 3 - 3 / math.sqrt(24))

    def test_eval_uses_fixed_statistics_and_still_differentiates(self):
        inputs = Tensor([1, 3], shape=(2, 1, 1, 1), requires_grad=True)
        weight, bias = Tensor([2], requires_grad=True), Tensor([1], requires_grad=True)
        running_mean, running_var = [2.0], [1.0]
        result = batch_norm2d(inputs, weight, bias, running_mean, running_var, train=False, eps=3)
        self.assertEqual(result.values, [0, 2])
        self.assertEqual(running_mean, [2])
        self.assertEqual(running_var, [1])
        result.backward(Tensor([3, -1], shape=result.shape))
        self.assertEqual(inputs.grad.values, [3, -1])
        self.assertEqual(weight.grad.values, [-2])
        self.assertEqual(bias.grad.values, [2])

    def test_old_graph_is_independent_of_later_running_statistics(self):
        inputs = Tensor([1, 3], shape=(2, 1, 1, 1), requires_grad=True)
        weight, bias = Tensor([2], requires_grad=True), Tensor([1], requires_grad=True)
        running_mean, running_var = [2.0], [1.0]
        old = batch_norm2d(inputs, weight, bias, running_mean, running_var, train=False, eps=3)
        batch_norm2d(Tensor([100, 200], shape=(2, 1, 1, 1)), weight, bias, running_mean, running_var, momentum=1)
        self.assertEqual(running_mean, [150])
        self.assertEqual(running_var, [5000])
        old.backward(Tensor([3, -1], shape=old.shape))
        self.assertEqual(inputs.grad.values, [3, -1])
        self.assertEqual(weight.grad.values, [-2])
        self.assertEqual(bias.grad.values, [2])

    def test_plain_input_updates_statistics_and_returns_plain_data(self):
        running_mean, running_var = [0.0], [1.0]
        result = batch_norm2d([[[[1, 3]]]], [1], [0], running_mean, running_var, momentum=1, eps=3)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [[[[-0.5, 0.5]]]])
        self.assertEqual(running_mean, [2])
        self.assertEqual(running_var, [2])

    def test_disabled_recording_does_not_disable_training_statistics(self):
        inputs = Tensor([1, 3], shape=(2, 1, 1, 1), requires_grad=True)
        running_mean, running_var = [0.0], [1.0]
        with set_recording(False):
            result = batch_norm2d(inputs, [1], [0], running_mean, running_var, momentum=1)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())
        self.assertTrue(inputs.requires_grad)
        self.assertEqual(running_mean, [2])
        self.assertEqual(running_var, [2])

    def test_constant_channel_with_small_epsilon(self):
        inputs = Tensor([7, 7], shape=(2, 1, 1, 1), requires_grad=True)
        weight, bias = Tensor([2], requires_grad=True), Tensor([3], requires_grad=True)
        running_mean, running_var = [0.0], [1.0]
        result = batch_norm2d(inputs, weight, bias, running_mean, running_var, momentum=1, eps=1e-12)
        self.assertEqual(result.values, [3, 3])
        self.assertEqual(running_var, [0])
        result.backward(Tensor([1, -1], shape=result.shape))
        self.assertEqual(inputs.grad.values, [2e6, -2e6])
        self.assertEqual(weight.grad.values, [0])
        self.assertEqual(bias.grad.values, [0])
        self.assertTrue(all(math.isfinite(value) for value in inputs.grad.values))

    def test_weight_or_bias_tensor_preserves_tensor_result(self):
        for weight, bias in ((Tensor([2], requires_grad=True), [1]), ([2], Tensor([1], requires_grad=True))):
            with self.subTest(weight=weight, bias=bias):
                result = batch_norm2d([[[[1, 3]]]], weight, bias, [2], [1], train=False, eps=3)
                self.assertIsInstance(result, Tensor)
                self.assertTrue(result.requires_grad)

    def test_empty_eval_batch_and_single_training_element(self):
        inputs = Tensor([], shape=(0, 2, 1, 1), requires_grad=True)
        weight, bias = Tensor([1, 1], requires_grad=True), Tensor([0, 0], requires_grad=True)
        result = batch_norm2d(inputs, weight, bias, [0, 0], [1, 1], train=False)
        self.assertEqual(result.shape, (0, 2, 1, 1))
        result.backward(Tensor([], shape=result.shape))
        self.assertEqual(inputs.grad.values, [])
        self.assertEqual(weight.grad.values, [0, 0])
        self.assertEqual(bias.grad.values, [0, 0])
        with self.assertRaisesRegex(ValueError, "至少两个"):
            batch_norm2d([[[[1]]]], [1], [0], [0], [1])

    def test_invalid_shapes_and_epsilon(self):
        cases = [
            ([1, 2], [1], [0], [0], [1], 1e-5),
            ([[[[1, 2]]]], [1, 2], [0], [0], [1], 1e-5),
            ([[[[1, 2]]]], [1], [0], [], [1], 1e-5),
            ([[[[1, 2]]]], [1], [0], [0], [1], 0),
        ]
        for inputs, weight, bias, average, variance, eps in cases:
            with self.subTest(inputs=inputs, weight=weight, eps=eps):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    batch_norm2d(inputs, weight, bias, average, variance, eps=eps)


if __name__ == "__main__":
    unittest.main()
