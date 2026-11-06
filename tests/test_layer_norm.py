"""最后一轴的层归一化、参数梯度和前向快照。"""

import math
import unittest
from math import prod

from qutai import Tensor
from qutai.core.context import set_recording, set_training
from qutai.core.tensor import layer_norm


class LayerNormTests(unittest.TestCase):
    def check_gradients(self, shape):
        width = shape[-1]
        inputs = Tensor([(index % 7 - 3) / 5 for index in range(prod(shape))], shape=shape, requires_grad=True)
        weight = Tensor([1.2, -0.7, 0.5], requires_grad=True)
        bias = Tensor([0.3, -0.2, 0.8], requires_grad=True)
        result = layer_norm(inputs, weight, bias, eps=0.2)
        seed = [(index % 5 - 2) / 3 for index in range(result.size)]
        result.backward(Tensor(seed, shape=shape))
        epsilon = 1e-6
        self.assertEqual(weight.size, width)
        for parent in (inputs, weight, bias):
            for index, analytic in enumerate(parent.grad.values):
                original = parent.values[index]
                parent.values[index] = original + epsilon
                upper = layer_norm(inputs, weight, bias, eps=0.2)
                parent.values[index] = original - epsilon
                lower = layer_norm(inputs, weight, bias, eps=0.2)
                parent.values[index] = original
                numeric = math.fsum((a - b) * incoming for a, b, incoming in zip(upper.values, lower.values, seed, strict=True)) / (2 * epsilon)
                self.assertAlmostEqual(analytic, numeric, places=7)

    def test_vector_and_three_dimensional_gradients_by_central_difference(self):
        for shape in ((3,), (2, 2, 3)):
            with self.subTest(shape=shape):
                self.check_gradients(shape)

    def test_population_variance_and_explicit_output_gradient(self):
        inputs = Tensor([1, 3], requires_grad=True)
        weight, bias = Tensor([2, 4], requires_grad=True), Tensor([1, 2], requires_grad=True)
        result = layer_norm(inputs, weight, bias, eps=3)
        self.assertEqual(result.tolist(), [0, 4])
        result.backward([3, -1])
        self.assertEqual(inputs.grad.tolist(), [1.875, -1.875])
        self.assertEqual(weight.grad.tolist(), [-1.5, -0.5])
        self.assertEqual(bias.grad.tolist(), [3, -1])

    def test_last_axis_statistics_are_independent_across_batches_and_positions(self):
        inputs = Tensor([[[1, 3], [101, 103]], [[1, 5], [100, 104]]], requires_grad=True)
        result = layer_norm(inputs, [2, -1], [1, 3], eps=3)
        self.assertEqual(result.tolist()[0], [[0, 2.5], [0, 2.5]])
        for row in result.tolist()[1]:
            self.assertAlmostEqual(row[0], 1 - 4 / math.sqrt(7))
            self.assertAlmostEqual(row[1], 3 - 2 / math.sqrt(7))
        result[0, 0].sum().backward()
        self.assertEqual(inputs.grad.values[2:], [0.0] * 6)
        self.assertNotEqual(inputs.grad.values[:2], [0, 0])

    def test_affine_parameter_gradients_accumulate_across_all_groups(self):
        inputs = Tensor([[[1, 3], [101, 103]], [[5, 7], [11, 13]]], requires_grad=True)
        weight, bias = Tensor([1, 1], requires_grad=True), Tensor([0, 0], requires_grad=True)
        layer_norm(inputs, weight, bias, eps=3).sum().backward()
        self.assertEqual(weight.grad.tolist(), [-2, 2])
        self.assertEqual(bias.grad.tolist(), [4, 4])
        self.assertEqual(inputs.grad.values, [0.0] * 8)

    def test_constant_features_and_small_epsilon(self):
        inputs = Tensor([[7, 7, 7]], requires_grad=True)
        weight, bias = Tensor([2, -1, 3], requires_grad=True), Tensor([1, 2, 3], requires_grad=True)
        result = layer_norm(inputs, weight, bias, eps=1e-12)
        self.assertEqual(result.tolist(), [[1, 2, 3]])
        result.backward([[1, -1, 0]])
        self.assertEqual(inputs.grad.tolist(), [[1e6, 0, -1e6]])
        self.assertEqual(weight.grad.tolist(), [0, 0, 0])
        self.assertEqual(bias.grad.tolist(), [1, -1, 0])
        self.assertTrue(all(math.isfinite(value) for value in inputs.grad.values))

    def test_single_feature_has_zero_input_and_weight_gradients(self):
        inputs = Tensor([[[1], [100]], [[-3], [9]]], requires_grad=True)
        weight, bias = Tensor([2], requires_grad=True), Tensor([3], requires_grad=True)
        result = layer_norm(inputs, weight, bias)
        self.assertEqual(result.tolist(), [[[3], [3]], [[3], [3]]])
        result.backward([[[1], [-2]], [[3], [4]]])
        self.assertEqual(inputs.grad.values, [0, 0, 0, 0])
        self.assertEqual(weight.grad.tolist(), [0])
        self.assertEqual(bias.grad.tolist(), [6])

    def test_empty_batches_keep_shape_and_return_zero_parameter_gradients(self):
        for shape in ((0, 3), (2, 0, 3)):
            with self.subTest(shape=shape):
                inputs = Tensor([], shape=shape, requires_grad=True)
                weight, bias = Tensor([1, 2, 3], requires_grad=True), Tensor([0, 0, 0], requires_grad=True)
                result = layer_norm(inputs, weight, bias)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.values, [])
                result.backward(Tensor([], shape=shape))
                self.assertEqual(inputs.grad.shape, shape)
                self.assertEqual(inputs.grad.values, [])
                self.assertEqual(weight.grad.tolist(), [0, 0, 0])
                self.assertEqual(bias.grad.tolist(), [0, 0, 0])

    def test_training_mode_does_not_change_math_or_disable_gradients(self):
        inputs = Tensor([[1, 3]], requires_grad=True)
        with set_training(True):
            training = layer_norm(inputs, [2, 4], [1, 2], eps=3)
        with set_training(False):
            evaluation = layer_norm(inputs, [2, 4], [1, 2], eps=3)
        self.assertEqual(training.tolist(), evaluation.tolist())
        self.assertTrue(evaluation.requires_grad)
        with set_recording(False):
            prediction = layer_norm(inputs, [2, 4], [1, 2], eps=3)
        self.assertEqual(prediction.tolist(), training.tolist())
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())

    def test_old_graph_uses_forward_snapshots_after_inputs_and_weights_change(self):
        inputs = Tensor([1, 3], requires_grad=True)
        weight, bias = Tensor([2, 4], requires_grad=True), Tensor([1, 2], requires_grad=True)
        old = layer_norm(inputs, weight, bias, eps=3)
        inputs.values[:] = [100, 200]
        weight.values[:] = [20, 40]
        bias.values[:] = [10, 20]
        old.values[:] = [999, 999]
        layer_norm(inputs, weight, bias)
        old.backward([3, -1])
        self.assertEqual(inputs.grad.tolist(), [1.875, -1.875])
        self.assertEqual(weight.grad.tolist(), [-1.5, -0.5])
        self.assertEqual(bias.grad.tolist(), [3, -1])

    def test_large_offsets_do_not_cancel_the_variance(self):
        result = layer_norm([1e12 + 1, 1e12 + 3], [1, 1], [0, 0], eps=3)
        self.assertEqual(result, [-0.5, 0.5])

    def test_plain_data_and_tensor_parameters_follow_existing_return_convention(self):
        self.assertEqual(layer_norm([[1, 3]], [2, 4], [1, 2], eps=3), [[0, 4]])
        for weight, bias in ((Tensor([2, 4], requires_grad=True), [1, 2]), ([2, 4], Tensor([1, 2], requires_grad=True))):
            result = layer_norm([[1, 3]], weight, bias, eps=3)
            self.assertIsInstance(result, Tensor)
            self.assertTrue(result.requires_grad)
            self.assertEqual(result.tolist(), [[0, 4]])

    def test_invalid_feature_shapes_and_epsilon(self):
        for data, weight, bias, eps in [
            (1, [1], [0], 1e-5),
            ([[], []], [], [], 1e-5),
            ([1, 2], [1], [0, 0], 1e-5),
            ([1, 2], [1, 1], [[0, 0]], 1e-5),
            ([1, 2], [1, 1], [0, 0], 0),
            ([1, 2], [1, 1], [0, 0], -1),
        ]:
            with self.subTest(data=data, eps=eps), self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                layer_norm(data, weight, bias, eps=eps)


if __name__ == "__main__":
    unittest.main()
