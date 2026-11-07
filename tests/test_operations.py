import unittest
from typing import get_type_hints

import qutai as qt
from qutai.core import tensor as tensor_module
from qutai.core.tensor import Tensor


class OperationsTests(unittest.TestCase):
    def test_public_functions_and_annotations(self):
        for name in (
            "add", "subtract", "multiply", "divide", "negate", "square",
            "relu", "matmul", "transpose", "sum", "mean", "reshape",
        ):
            function = getattr(qt, name)
            self.assertIs(function, getattr(tensor_module, name))
            self.assertIn("return", get_type_hints(function))

    def test_broadcast_arithmetic(self):
        self.assertEqual(qt.add([[1, 2, 3], [4, 5, 6]], [10, 20, 30]),
                         [[11, 22, 33], [14, 25, 36]])
        self.assertEqual(qt.multiply([[1], [2]], [10, 20, 30]),
                         [[10, 20, 30], [20, 40, 60]])
        self.assertEqual(qt.subtract(10, [1, 2]), [9, 8])
        self.assertEqual(qt.divide([3, 8], [2, 4]), [1.5, 2])
        self.assertEqual(qt.add([[[1, 2]], [[3, 4]]], [10, 20]),
                         [[[11, 22]], [[13, 24]]])
        with self.assertRaises(ValueError):
            qt.add([[1, 2, 3]], [1, 2])

    def test_unary(self):
        self.assertEqual(qt.negate([1, -2]), [-1, 2])
        self.assertEqual(qt.square([[1, -2], [3, -4]]), [[1, 4], [9, 16]])
        self.assertEqual(qt.relu([-2, 0, 3]), [0, 0, 3])

    def test_matrix_and_transforms(self):
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(qt.matmul(data, [[7, 8], [9, 10], [11, 12]]),
                         [[58, 64], [139, 154]])
        self.assertEqual(qt.transpose(data), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(qt.transpose(qt.transpose(data)), data)
        self.assertEqual(qt.reshape(data, (-1, 2)), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(qt.reshape([[5]], ()), 5)
        self.assertEqual(qt.matmul([1, 2], [[3], [4]]), [11])
        self.assertEqual(qt.matmul([1, 2], [3, 4]), 11)
        for left, right in [([1, 2], [[3], [4], [5]]), ([[1, 2]], [[3, 4]])]:
            with self.assertRaisesRegex(ValueError, "矩阵"):
                qt.matmul(left, right)
        self.assertEqual(qt.transpose([1, 2]), [1, 2])
        with self.assertRaises(ValueError):
            qt.transpose(data, axes=(0, 0))
        with self.assertRaises(ValueError):
            qt.reshape(data, (-1, -1))
        with self.assertRaises(ValueError):
            qt.reshape(data, (4, -1))

    def test_high_dimensional_transpose(self):
        tensor = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        result = qt.transpose(tensor, axes=(0, 2, 1))
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertEqual(result.tolist(),
                         [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]])
        self.assertEqual(qt.transpose(tensor).shape, (3, 2, 2))

    def test_reductions(self):
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(qt.sum(data), 21)
        self.assertEqual(qt.sum(data, axis=0), [5, 7, 9])
        self.assertEqual(qt.sum(data, axis=-1, keepdims=True), [[6], [15]])
        self.assertEqual(qt.mean(data), 3.5)
        self.assertEqual(qt.mean(data, axis=0), [2.5, 3.5, 4.5])
        self.assertEqual(qt.mean(data, axis=1, keepdims=True), [[2], [5]])
        self.assertEqual(qt.sum(data, keepdims=True), [[21]])
        self.assertEqual(qt.mean(5, keepdims=True), 5)
        with self.assertRaises(ValueError):
            qt.sum(data, axis=2)

    def test_empty_dimensions(self):
        self.assertEqual(qt.sum([]), 0)
        self.assertEqual(qt.sum([[], []], axis=1, keepdims=True), [[0], [0]])
        self.assertEqual(qt.mean([[], []], axis=0), [])
        for data, axis in [([], None), ([[], []], 1)]:
            with self.assertRaisesRegex(ValueError, "无法求平均值"):
                qt.mean(data, axis=axis)
        empty = qt.reshape(Tensor([]), (0, 3))
        self.assertEqual(empty.shape, (0, 3))
        self.assertEqual(qt.transpose(empty).shape, (3, 0))

    def test_tensor_results_and_input_ownership(self):
        tensor = Tensor([[1, 2], [3, 4]])
        result = qt.add(tensor, 1)
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[2, 3], [4, 5]])
        self.assertEqual(tensor.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(qt.mean(tensor, axis=1, keepdims=True).shape, (2, 1))
        self.assertEqual(qt.reshape(tensor, (-1,)).shape, (4,))
        self.assertEqual(qt.sum(tensor).item(), 10)


if __name__ == "__main__":
    unittest.main()
