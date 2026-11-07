import random
import unittest

import qutai as qt
from qutai import Tensor, nn


class ReLUTests(unittest.TestCase):
    def test_forward_gradients_and_no_parameters(self):
        layer = nn.ReLU()
        result = layer([[-2, 0], [3, -1]])
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.tolist(), [[0, 0], [3, 0]])
        self.assertFalse(result.requires_grad)
        self.assertEqual(layer.parameters(), ())
        inputs = Tensor([-2, 0, 3], requires_grad=True)
        qt.sum(layer(inputs)).backward()
        self.assertEqual(inputs.grad.tolist(), [0, 0, 1])
        self.assertEqual(inputs.tolist(), [-2, 0, 3])


class LinearTests(unittest.TestCase):
    def test_known_multi_output_forward(self):
        layer = nn.Linear(2, 3, seed=42)
        layer.weight.values = [1, 2, 3, 4, 5, 6]
        layer.bias.values = [0.5, -0.5, 1]
        inputs = [[1, 2], [-1, 0]]
        for data in (inputs, Tensor(inputs)):
            result = layer(data)
            self.assertIsInstance(result, Tensor)
            self.assertEqual(result.shape, (2, 3))
            self.assertEqual(result.tolist(), [[9.5, 11.5, 16], [-0.5, -2.5, -2]])
            self.assertTrue(result.requires_grad)

    def test_input_weight_and_bias_gradients(self):
        layer = nn.Linear(2, 2, seed=42)
        layer.weight.values = [1, 2, 3, 4]
        layer.bias.values = [0.5, -0.5]
        inputs = Tensor([[1, 2], [3, 4]], requires_grad=True)
        qt.sum(layer(inputs)).backward()
        self.assertEqual(inputs.grad.tolist(), [[3, 7], [3, 7]])
        self.assertEqual(layer.weight.grad.tolist(), [[4, 4], [6, 6]])
        self.assertEqual(layer.bias.grad.tolist(), [2, 2])

    def test_reproducible_initialization_and_parameter_references(self):
        state = random.getstate()
        first = nn.Linear(2, 3, seed=42)
        second = nn.Linear(2, 3, seed=42)
        other = nn.Linear(2, 3, seed=7)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first.weight.tolist(), second.weight.tolist())
        self.assertNotEqual(first.weight.tolist(), other.weight.tolist())
        self.assertEqual(first.weight.shape, (2, 3))
        self.assertEqual(first.bias.tolist(), [0, 0, 0])
        self.assertTrue(all(-2 ** -0.5 <= value <= 2 ** -0.5 for value in first.weight.values))
        parameters = first.parameters()
        self.assertIs(parameters[0], first.weight)
        self.assertIs(parameters[1], first.bias)
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertTrue(all(parameter.parents == () for parameter in parameters))

    def test_feature_counts_and_input_shapes(self):
        for in_features, out_features in ((0, 1), (-1, 1), (2, 0), (2, -1)):
            with self.subTest(in_features=in_features, out_features=out_features):
                with self.assertRaisesRegex(ValueError, "特征数必须大于零"):
                    nn.Linear(in_features, out_features)
        layer = nn.Linear(2, 1, seed=42)
        layer.weight.values = [2, 3]
        layer.bias.values = [0.5]
        self.assertEqual(layer([1, 2]).shape, (1,))
        self.assertEqual(layer([1, 2]).tolist(), [8.5])
        for inputs in ([1, 2, 3], [[1, 2, 3]]):
            with self.subTest(inputs=inputs):
                with self.assertRaisesRegex(ValueError, "矩阵"):
                    layer(inputs)


if __name__ == "__main__":
    unittest.main()
