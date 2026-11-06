"""反向传播：手算、有限差分、共享图、广播和逐元素链式计算。"""

import unittest
from dataclasses import dataclass
from typing import get_type_hints

import qutai as qt
from qutai.core.grad import backward, record
from qutai.backends.device import get_backend
from qutai.core.tensor import Tensor
from qutai.core.types import Backward


@dataclass(eq=False)
class Node:
    """仅有计算图字段，用于验证引擎不依赖 Tensor。"""
    requires_grad: bool = True
    parents: tuple[Node, ...] = ()
    backward_rule: Backward | None = None


class GradientTests(unittest.TestCase):
    def test_engine_without_tensor(self):
        x = Node()
        shared = record(Node(), (x, x), lambda gradient: (gradient, gradient))
        output = record(Node(), (shared, shared), lambda gradient: (gradient, gradient))
        self.assertEqual(backward(output, get_backend().storage([2.0, -3.0])), {x: [8.0, -12.0]})

    def test_add_sum_backward(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        loss = qt.sum(qt.add(x, x))
        self.assertEqual(loss.item(), 12)
        loss.backward()
        self.assertEqual(x.grad.tolist(), [2, 2, 2])
        self.assertFalse(x.grad.requires_grad)
        self.assertIsNone(loss.grad)

    def test_shared_graph_and_accumulation(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        shared = qt.add(x, x)
        loss = qt.sum(qt.add(shared, shared))
        loss.backward()
        self.assertEqual(x.grad.tolist(), [4, 4, 4])
        loss.backward()
        self.assertEqual(x.grad.tolist(), [8, 8, 8])
        x.zero_grad()
        self.assertIsNone(x.grad)
        qt.sum(qt.add(x, 1)).backward()
        self.assertEqual(x.grad.tolist(), [1, 1, 1])

    def test_broadcast_bias(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        bias = Tensor([10, 20, 30], requires_grad=True)
        scalar = Tensor(2, requires_grad=True)
        qt.sum(qt.add(qt.add(x, bias), scalar)).backward()
        self.assertEqual(x.grad.tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(bias.grad.tolist(), [2, 2, 2])
        self.assertEqual(scalar.grad.item(), 6)

    def test_high_dimensional_broadcast(self):
        x = Tensor([[[1, 2, 3]], [[4, 5, 6]]], requires_grad=True)
        bias = Tensor([[[10], [20], [30], [40]]], requires_grad=True)
        qt.sum(qt.add(x, bias)).backward()
        self.assertEqual(x.grad.tolist(), [[[4, 4, 4]], [[4, 4, 4]]])
        self.assertEqual(bias.grad.tolist(), [[[6], [6], [6], [6]]])

    def test_axis_keepdims_and_explicit_seed(self):
        for keepdims in (False, True):
            x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
            result = qt.sum(x, axis=-1, keepdims=keepdims)
            result.backward([[2], [3]] if keepdims else [2, 3])
            self.assertEqual(x.grad.tolist(), [[2, 2, 2], [3, 3, 3]])
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        qt.sum(x, axis=0).backward([1, 2, 3])
        self.assertEqual(x.grad.tolist(), [[1, 2, 3], [1, 2, 3]])
        x.zero_grad()
        qt.sum(x, keepdims=True).backward()
        self.assertEqual(x.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

    def test_constant_parent(self):
        x = Tensor([1, 2], requires_grad=True)
        constant = Tensor([3, 4])
        qt.sum(qt.add(x, constant)).backward()
        self.assertEqual(x.grad.tolist(), [1, 1])
        self.assertIsNone(constant.grad)

    def test_empty_broadcast(self):
        x = Tensor([], shape=(0, 3), requires_grad=True)
        bias = Tensor([1, 2, 3], requires_grad=True)
        scalar = Tensor(1, requires_grad=True)
        qt.sum(qt.add(qt.add(x, bias), scalar)).backward()
        self.assertEqual(x.grad.shape, (0, 3))
        self.assertEqual(x.grad.tolist(), [])
        self.assertEqual(bias.grad.tolist(), [0, 0, 0])
        self.assertEqual(scalar.grad.item(), 0)

    def test_empty_axis(self):
        x = Tensor([[], []], requires_grad=True)
        qt.sum(x, axis=1).backward([2, 3])
        self.assertEqual(x.grad.shape, (2, 0))
        self.assertEqual(x.grad.tolist(), [[], []])

    def test_finite_differences(self):
        rows = [[0.2, -0.4, 0.7], [0.3, 0.1, -0.6]]
        bias = [0.5, -0.2, 0.8]

        def objective(a, b):
            return qt.sum(qt.add(qt.add(a, b), qt.add(a, [[0.1], [0.2]])))

        x = Tensor(rows, requires_grad=True)
        b = Tensor(bias, requires_grad=True)
        objective(x, b).backward()
        epsilon = 1e-6
        for row in range(2):
            for column in range(3):
                high, low = [r[:] for r in rows], [r[:] for r in rows]
                high[row][column] += epsilon
                low[row][column] -= epsilon
                numeric = (objective(high, bias) - objective(low, bias)) / (2 * epsilon)
                self.assertAlmostEqual(x.grad.tolist()[row][column], numeric, places=7)
        for index in range(3):
            high, low = bias[:], bias[:]
            high[index] += epsilon
            low[index] -= epsilon
            numeric = (objective(rows, high) - objective(rows, low)) / (2 * epsilon)
            self.assertAlmostEqual(b.grad.tolist()[index], numeric, places=7)

    def test_missing_rule_fails_before_updating_gradients(self):
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        unknown = record(Tensor(x.tolist()), (x,))
        with self.assertRaisesRegex(NotImplementedError, "尚未实现反向规则"):
            qt.sum(qt.add(x, unknown)).backward()
        self.assertIsNone(x.grad)

    def test_square_sum_backward(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        loss = qt.sum(qt.square(x))
        self.assertEqual(loss.item(), 14)
        loss.backward()
        self.assertEqual(x.grad.tolist(), [2, 4, 6])

    def test_unary_explicit_seeds(self):
        cases = [
            (qt.square, [1, -2, 0], [0.5, 3, -1], [1, -12, 0]),
            (qt.negate, [1, -2, 0], [2, -3, 0.5], [-2, 3, -0.5]),
            (qt.relu, [-2, 0, 3], [2, 3, 4], [0, 0, 4]),
        ]
        for operation, values, seed, expected in cases:
            with self.subTest(operation=operation.__name__):
                x = Tensor(values, requires_grad=True)
                operation(x).backward(seed)
                self.assertEqual(x.grad.tolist(), expected)

    def test_unary_chain_and_shared_paths(self):
        x = Tensor([-2, -0.5, 0.75, 3], requires_grad=True)
        loss = qt.sum(qt.square(qt.relu(qt.negate(x))))
        self.assertEqual(loss.item(), 4.25)
        loss.backward()
        self.assertEqual(x.grad.tolist(), [-4, -1, 0, 0])
        x.zero_grad()
        shared = qt.square(x)
        qt.sum(qt.add(shared, shared)).backward()
        self.assertEqual(x.grad.tolist(), [-8, -2, 3, 12])

    def test_unary_finite_differences_with_broadcast(self):
        rows = [[-1.2, 0.7], [0.5, -0.9]]
        bias = [0.1, -0.2]

        def objective(a, b):
            return qt.sum(qt.square(qt.relu(qt.add(a, b))))

        x = Tensor(rows, requires_grad=True)
        b = Tensor(bias, requires_grad=True)
        objective(x, b).backward()
        epsilon = 1e-6
        for row in range(2):
            for column in range(2):
                high, low = [r[:] for r in rows], [r[:] for r in rows]
                high[row][column] += epsilon
                low[row][column] -= epsilon
                numeric = (objective(high, bias) - objective(low, bias)) / (2 * epsilon)
                self.assertAlmostEqual(x.grad.tolist()[row][column], numeric, places=7)
        for index in range(2):
            high, low = bias[:], bias[:]
            high[index] += epsilon
            low[index] -= epsilon
            numeric = (objective(rows, high) - objective(rows, low)) / (2 * epsilon)
            self.assertAlmostEqual(b.grad.tolist()[index], numeric, places=7)

    def test_negate_finite_differences(self):
        values = [-1.3, 0.4, 2.1]
        x = Tensor(values, requires_grad=True)
        qt.sum(qt.square(qt.negate(x))).backward()
        epsilon = 1e-6
        for index in range(3):
            high, low = values[:], values[:]
            high[index] += epsilon
            low[index] -= epsilon
            numeric = (qt.sum(qt.square(qt.negate(high)))
                       - qt.sum(qt.square(qt.negate(low)))) / (2 * epsilon)
            self.assertAlmostEqual(x.grad.tolist()[index], numeric, places=7)

    def test_unary_scalar_and_empty(self):
        x = Tensor(3, requires_grad=True)
        qt.square(x).backward()
        self.assertEqual(x.grad.item(), 6)
        zero = Tensor(0, requires_grad=True)
        qt.relu(zero).backward()
        self.assertEqual(zero.grad.item(), 0)
        for operation in (qt.negate, qt.square, qt.relu):
            empty = Tensor([], shape=(0, 3), requires_grad=True)
            qt.sum(operation(empty)).backward()
            self.assertEqual(empty.grad.shape, (0, 3))
            self.assertEqual(empty.grad.tolist(), [])
            constant = operation(Tensor([1, 2]))
            self.assertFalse(constant.requires_grad)
            self.assertEqual(constant.parents, ())

    def test_seed_and_recording_errors(self):
        with self.assertRaises(ValueError):
            Tensor(1).backward()
        x = Tensor([1, 2], requires_grad=True)
        with self.assertRaises(ValueError):
            x.backward()
        with self.assertRaises(ValueError):
            x.backward(1)
        self.assertIsNone(x.grad)

    def test_leaf_and_long_chain(self):
        x = Tensor(1, requires_grad=True)
        x.backward()
        self.assertEqual(x.grad.item(), 1)
        x.zero_grad()
        result = x
        for _ in range(1500):
            result = qt.add(result, 1)
        qt.sum(result).backward()
        self.assertEqual(x.grad.item(), 1)

    def test_annotations(self):
        for function in (Tensor.backward, backward, record):
            self.assertIn("return", get_type_hints(function))


if __name__ == "__main__":
    unittest.main()
