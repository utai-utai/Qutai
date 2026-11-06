"""分类所需运算的数值、梯度、整数输出与模型读写。"""

import math as scalar_math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import qutai as qt
from qutai import Tensor, nn


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.loss_function = nn.CrossEntropy()

    def check_gradient(self, operation, values, shape, seed):
        """用独立的中心差分验证任意输出梯度下的反向结果。"""
        tensor = Tensor(values, shape=shape, requires_grad=True)
        output = operation(tensor)
        output.backward(Tensor(seed, shape=output.shape))
        for index, analytic in enumerate(tensor.grad.values):
            high, low = Tensor(values, shape=shape), Tensor(values, shape=shape)
            high.values[index] += 1e-6
            low.values[index] -= 1e-6
            upper = sum(value * weight for value, weight in zip(operation(high).values, seed, strict=True))
            lower = sum(value * weight for value, weight in zip(operation(low).values, seed, strict=True))
            self.assertAlmostEqual(analytic, (upper - lower) / 2e-6, places=7)

    def test_exp_log_values_and_input_output_types(self):
        self.assertEqual(qt.exp(0), 1.0)
        self.assertEqual(qt.log(1), 0.0)
        self.assertEqual(qt.exp(((0, 0), (0, 0))), [[1, 1], [1, 1]])
        self.assertAlmostEqual(qt.log(scalar_math.e), 1)
        for operation in (qt.exp, qt.log):
            empty = Tensor([], shape=(0, 3), requires_grad=True)
            output = operation(empty)
            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.shape, (0, 3))
            qt.sum(output).backward()
            self.assertEqual(empty.grad.shape, (0, 3))
            self.assertEqual(empty.grad.values, [])
            self.assertEqual(operation([]), [])

    def test_exp_log_weighted_gradients(self):
        self.check_gradient(qt.exp, [-0.8, 0.3, 1.2, -0.1], (2, 2), [0.2, -0.7, 1.3, 0.6])
        self.check_gradient(qt.log, [0.4, 0.9, 1.2, 2.5], (2, 2), [0.2, -0.7, 1.3, 0.6])

    def test_exp_log_chain_shared_paths_and_log_domain(self):
        tensor = Tensor([-1.2, 0.3, 1.5], requires_grad=True)
        shared = qt.log(qt.exp(tensor))
        qt.sum(qt.add(shared, shared)).backward()
        for value in tensor.grad.values:
            self.assertAlmostEqual(value, 2)
        for data in (0, -1, [1, 0], Tensor([1, -2])):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    qt.log(data)

    def test_argmax_nd_axes_first_tie_and_integer_results(self):
        data = [[[1, 8], [5, 4], [5, 9]], [[7, 8], [3, 6], [2, 1]]]
        tensor = Tensor(data, requires_grad=True)
        self.assertEqual(qt.argmax(tensor), 5)
        self.assertEqual(qt.argmax(data, axis=0), [[1, 0], [0, 1], [0, 0]])
        self.assertEqual(qt.argmax(tensor, axis=1), [[1, 2], [0, 0]])
        indices = qt.argmax(tensor, axis=-1)
        self.assertEqual(indices, [[1, 0, 1], [1, 1, 0]])
        self.assertTrue(all(type(index) is int for row in indices for index in row))
        self.assertIsNone(tensor.grad)
        self.assertTrue(tensor.requires_grad)

    def test_argmax_keepdims_and_scalar(self):
        data = [[[1, 8], [5, 4], [5, 9]], [[7, 8], [3, 6], [2, 1]]]
        self.assertEqual(qt.argmax(data, axis=1, keepdims=True), [[[1, 2]], [[0, 0]]])
        self.assertEqual(qt.argmax(data, keepdims=True), [[[5]]])
        self.assertEqual(qt.argmax([3, 3, 1]), 0)
        self.assertIs(type(qt.argmax(7)), int)
        self.assertEqual(qt.argmax(7, keepdims=True), 0)

    def test_argmax_empty_dimensions_and_invalid_axes(self):
        self.assertEqual(qt.argmax(Tensor([], shape=(0, 3)), axis=1), [])
        self.assertEqual(qt.argmax([[], []], axis=0), [])
        for data, axis in (([], None), ([[], []], 1), ([1, 2], 1), ([1, 2], -2), (1, 0)):
            with self.subTest(data=data, axis=axis):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    qt.argmax(data, axis=axis)

    def test_softmax_probabilities_and_large_offsets(self):
        layer = nn.Softmax()
        self.assertEqual(layer.parameters(), ())
        result = layer([0, scalar_math.log(2), scalar_math.log(3)])
        self.assertIsInstance(result, Tensor)
        for actual, expected in zip(result.values, (1 / 6, 2 / 6, 3 / 6), strict=True):
            self.assertAlmostEqual(actual, expected)
        scores = [[1000, 1001, 1002], [-1000, -1000, -1000], [1e20, 1e20, 1e20]]
        result = layer(scores)
        denominator = 1 + scalar_math.e + scalar_math.e ** 2
        expected = [1 / denominator, scalar_math.e / denominator, scalar_math.e ** 2 / denominator]
        for actual, reference in zip(result.tolist()[0], expected, strict=True):
            self.assertAlmostEqual(actual, reference)
        for row in result.tolist():
            self.assertAlmostEqual(sum(row), 1)
            self.assertTrue(all(0 <= value <= 1 for value in row))
        self.assertEqual(result.tolist()[1], result.tolist()[2])

    def test_softmax_nd_weighted_gradients_for_each_axis(self):
        values = [-0.8, 0.3, 1.2, -0.1, 0.4, 0.9, -0.4, 0.7, 0.2, -0.6, 0.5, 1.1]
        seed = [0.2, -0.7, 1.3, 0.6, -0.3, 0.8, 0.4, -0.5, 1.1, 0.9, -0.2, 0.1]
        for axis in (0, 1, -1):
            with self.subTest(axis=axis):
                self.check_gradient(nn.Softmax(axis=axis), values, (2, 3, 2), seed)

    def test_softmax_empty_dimensions_and_invalid_axes(self):
        for shape, axis in (((0, 3), 1), ((2, 0), 0)):
            tensor = Tensor([], shape=shape, requires_grad=True)
            result = nn.Softmax(axis=axis)(tensor)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.values, [])
            qt.sum(result).backward()
            self.assertEqual(tensor.grad.shape, shape)
        for data, axis in ((1, -1), ([], -1), ([[], []], 1), ([1, 2], 1), ([1, 2], -2)):
            with self.subTest(data=data, axis=axis):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    nn.Softmax(axis=axis)(data)

    def test_softmax_single_category_has_zero_gradient(self):
        tensor = Tensor([[2000], [-2000]], requires_grad=True)
        result = nn.Softmax()(tensor)
        self.assertEqual(result.tolist(), [[1], [1]])
        result.backward([[3], [-2]])
        self.assertEqual(tensor.grad.tolist(), [[0], [0]])

    def test_cross_entropy_known_probabilities_and_scaled_gradient(self):
        scores = Tensor([[scalar_math.log(2), scalar_math.log(3), scalar_math.log(5)], [0, 0, 0]], requires_grad=True)
        loss = self.loss_function(scores, (0, 2))
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(loss.item(), (scalar_math.log(5) + scalar_math.log(3)) / 2)
        loss.backward(2.5)
        expected = [-1, 0.375, 0.625, 5 / 12, 5 / 12, -5 / 6]
        for actual, reference in zip(scores.grad.values, expected, strict=True):
            self.assertAlmostEqual(actual, reference)
        constant = self.loss_function([[0, 0, 0]], [1])
        self.assertIsInstance(constant, Tensor)
        self.assertFalse(constant.requires_grad)
        self.assertAlmostEqual(constant.item(), scalar_math.log(3))

    def test_cross_entropy_finite_differences(self):
        self.check_gradient(lambda scores: self.loss_function(scores, [2, 0]), [-0.8, 0.3, 1.2, -0.1, 0.4, 0.9], (2, 3), [-1.7])

    def test_cross_entropy_extreme_scores_and_common_offset(self):
        self.assertEqual(self.loss_function([[0, -1e308], [0, -1e308]], [1, 1]).item(), 1e308)
        scores = Tensor([[1000, -1000], [-1000, 1000]], requires_grad=True)
        loss = self.loss_function(scores, [1, 0])
        self.assertEqual(loss.item(), 2000)
        loss.backward()
        self.assertEqual(scores.grad.tolist(), [[0.5, -0.5], [-0.5, 0.5]])
        scores = Tensor([[1e20, 1e20, 1e20]], requires_grad=True)
        loss = self.loss_function(scores, [1])
        self.assertAlmostEqual(loss.item(), scalar_math.log(3))
        loss.backward()
        for actual, expected in zip(scores.grad.values, (1 / 3, -2 / 3, 1 / 3), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_cross_entropy_shape_and_class_ids(self):
        cases = [
            ([1, 2], [0]),
            ([[[1, 2]]], [0]),
            (Tensor([], shape=(0, 3)), []),
            ([[], []], [0, 0]),
            ([[1, 2], [3, 4]], [0]),
            ([[1, 2]], [-1]),
            ([[1, 2]], [2]),
            ([[1, 2]], [0.5]),
        ]
        for scores, labels in cases:
            with self.subTest(scores=scores, labels=labels):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    self.loss_function(scores, labels)

    def test_softmax_serialization_keeps_axis_and_predictions(self):
        source = nn.Model([nn.Linear(2, 3, seed=42), nn.Model([nn.Softmax(axis=0)])])
        inputs = [[1, 2], [-1, 0]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model().load(path)
        layer = restored.layers[1].layers[0]
        self.assertIsInstance(layer, nn.Softmax)
        self.assertEqual(layer.axis, 0)
        self.assertEqual(layer.parameters(), ())
        prediction = restored(inputs, train=False)
        self.assertEqual(prediction.tolist(), source(inputs, train=False).tolist())
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        qt.sum(qt.square(restored(inputs))).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in restored.parameters()))

    def test_cross_entropy_respects_inference_mode(self):
        model = nn.Model([nn.Linear(2, 3, seed=42)])
        scores = model([[1, 2], [-1, 0]], train=False)
        loss = self.loss_function(scores, [0, 2])
        self.assertFalse(loss.requires_grad)
        self.assertEqual(loss.parents, ())
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.loss_function(model([[1, 2], [-1, 0]]), [0, 2]).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
