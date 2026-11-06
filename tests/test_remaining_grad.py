"""验证二元、归约、形状变换与二维矩阵乘法的反向规则。"""

import unittest
from collections.abc import Callable

import qutai as qt
from qutai import Tensor
from qutai.core.types import Data, Result


type Binary = Callable[[Data | Tensor, Data | Tensor], Result | Tensor]


class RemainingGradientTests(unittest.TestCase):
    def check_binary_gradient(self, operation: Binary, left: Data, right: Data) -> None:
        a = Tensor(left, requires_grad=True)
        b = Tensor(right, requires_grad=True)
        qt.mean(qt.square(operation(a, b))).backward()
        epsilon = 1e-6

        for index, parent in enumerate((a, b)):
            for position, analytic in enumerate(parent.grad.values):
                high = [Tensor(left), Tensor(right)]
                low = [Tensor(left), Tensor(right)]
                high[index].values[position] += epsilon
                low[index].values[position] -= epsilon
                upper = qt.mean(qt.square(operation(*high))).item()
                lower = qt.mean(qt.square(operation(*low))).item()
                self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=7)

    def test_broadcast_binary_finite_differences(self):
        cases = [
            ([[[0.2, -0.4, 0.7]], [[0.3, 0.1, -0.6]]], [[[0.8], [1.3]]]),
            (0.7, [[0.8, -1.1], [1.3, 0.9]]),
            ([[0.2, -0.4], [0.3, 0.1]], 1.3),
        ]
        for operation in (qt.subtract, qt.multiply, qt.divide):
            for left, right in cases:
                with self.subTest(operation=operation.__name__, left=left):
                    self.check_binary_gradient(operation, left, right)

    def test_binary_hand_calculations(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([0.5, 1, 1.5], requires_grad=True)
        qt.sum(qt.subtract(a, b)).backward()
        self.assertEqual(a.grad.tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(b.grad.tolist(), [-2, -2, -2])

        a = Tensor([[1], [2]], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)
        qt.sum(qt.multiply(a, b)).backward()
        self.assertEqual(a.grad.tolist(), [[12], [12]])
        self.assertEqual(b.grad.tolist(), [3, 3, 3])

        a = Tensor([[2, 4], [6, 8]], requires_grad=True)
        b = Tensor([2, 4], requires_grad=True)
        qt.sum(qt.divide(a, b)).backward()
        self.assertEqual(a.grad.tolist(), [[0.5, 0.25], [0.5, 0.25]])
        self.assertEqual(b.grad.tolist(), [-2, -0.75])

    def test_shared_binary_inputs(self):
        for operation, expected in [
            (qt.subtract, [0, 0, 0]),
            (qt.multiply, [-4, 1, 6]),
            (qt.divide, [0, 0, 0]),
        ]:
            x = Tensor([-2, 0.5, 3], requires_grad=True)
            qt.sum(operation(x, x)).backward()
            self.assertEqual(x.grad.tolist(), expected)

    def test_mean_axes_and_keepdims(self):
        for axis, count in [(None, 6), (0, 2), (1, 3), (-1, 3)]:
            for keepdims in (False, True):
                x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
                qt.sum(qt.mean(x, axis=axis, keepdims=keepdims)).backward()
                self.assertEqual(x.grad.tolist(), [[1 / count] * 3, [1 / count] * 3])
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        qt.mean(x, axis=1).backward([2, 3])
        self.assertEqual(x.grad.tolist(), [[2 / 3] * 3, [1] * 3])

    def test_mean_finite_differences(self):
        data = [[[0.2, -0.4], [0.7, 0.1]], [[-0.3, 0.8], [0.5, -0.6]]]
        for axis in (None, 0, 1, -1):
            for keepdims in (False, True):
                x = Tensor(data, requires_grad=True)
                qt.sum(qt.square(qt.mean(x, axis, keepdims))).backward()
                for position, analytic in enumerate(x.grad.values):
                    high, low = Tensor(data), Tensor(data)
                    high.values[position] += 1e-6
                    low.values[position] -= 1e-6
                    upper = qt.sum(qt.square(qt.mean(high, axis, keepdims))).item()
                    lower = qt.sum(qt.square(qt.mean(low, axis, keepdims))).item()
                    self.assertAlmostEqual(analytic, (upper - lower) / 2e-6, places=7)

    def test_reshape_explicit_seed_and_shared_paths(self):
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        qt.reshape(x, (-1,)).backward([1, 2, 3, 4, 5, 6])
        self.assertEqual(x.grad.tolist(), [[1, 2, 3], [4, 5, 6]])
        x.zero_grad()
        qt.add(qt.sum(qt.reshape(x, (3, 2))), qt.sum(x)).backward()
        self.assertEqual(x.grad.tolist(), [[2, 2, 2], [2, 2, 2]])

    def test_transpose_non_self_inverse(self):
        x = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], requires_grad=True)
        result = qt.transpose(x, axes=(1, 2, 0))
        result.backward(Tensor(list(range(1, 13)), shape=result.shape))
        self.assertEqual(x.grad.tolist(),
                         [[[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]])

    def test_four_dimensional_transpose_finite_differences(self):
        values = [i / 10 for i in range(24)]
        x = Tensor(values, shape=(2, 1, 3, 4), requires_grad=True)
        qt.mean(qt.square(qt.transpose(x, axes=(-2, 0, -1, 1)))).backward()
        for position, analytic in enumerate(x.grad.values):
            high, low = Tensor(values, shape=x.shape), Tensor(values, shape=x.shape)
            high.values[position] += 1e-6
            low.values[position] -= 1e-6
            upper = qt.mean(qt.square(qt.transpose(high, axes=(-2, 0, -1, 1)))).item()
            lower = qt.mean(qt.square(qt.transpose(low, axes=(-2, 0, -1, 1)))).item()
            self.assertAlmostEqual(analytic, (upper - lower) / 2e-6, places=7)

    def test_matmul_hand_calculation(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([[7, 8], [9, 10], [11, 12]], requires_grad=True)
        qt.sum(qt.matmul(a, b)).backward()
        self.assertEqual(a.grad.tolist(), [[15, 19, 23], [15, 19, 23]])
        self.assertEqual(b.grad.tolist(), [[5, 5], [7, 7], [9, 9]])

    def test_matmul_finite_differences(self):
        self.check_binary_gradient(
            qt.matmul,
            [[0.2, -0.4, 0.7], [0.3, 0.1, -0.6]],
            [[0.8, -0.2], [0.5, 0.9], [-0.3, 0.4]],
        )

    def test_matmul_shared_input(self):
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        qt.sum(qt.matmul(x, x)).backward()
        self.assertEqual(x.grad.tolist(), [[7, 11], [9, 13]])

    def test_empty_axes_and_zero_inner_dimension(self):
        a = Tensor([[], []], requires_grad=True)
        b = Tensor([], shape=(0, 3), requires_grad=True)
        qt.sum(qt.matmul(a, b)).backward()
        self.assertEqual(a.grad.shape, (2, 0))
        self.assertEqual(a.grad.tolist(), [[], []])
        self.assertEqual(b.grad.shape, (0, 3))

        empty = Tensor([], shape=(0, 2), requires_grad=True)
        weights = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        qt.sum(qt.matmul(empty, weights)).backward()
        self.assertEqual(weights.grad.tolist(), [[0, 0, 0], [0, 0, 0]])

        x = Tensor([[], []], requires_grad=True)
        qt.sum(qt.mean(x, axis=0, keepdims=True)).backward()
        self.assertEqual(x.grad.shape, (2, 0))
        self.assertEqual(x.grad.tolist(), [[], []])

        x = Tensor([], shape=(0, 3), requires_grad=True)
        qt.sum(qt.transpose(qt.reshape(x, (3, 0)))).backward()
        self.assertEqual(x.grad.shape, (0, 3))

    def test_scalar_transforms(self):
        x = Tensor(2, requires_grad=True)
        qt.mean(qt.transpose(qt.reshape(x, (1, 1)))).backward()
        self.assertEqual(x.grad.item(), 1)


if __name__ == "__main__":
    unittest.main()
