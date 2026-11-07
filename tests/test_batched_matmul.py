import unittest
from math import prod

import qutai as qt
from qutai import Tensor
from qutai.backends import cpu


class BatchedMatmulTests(unittest.TestCase):
    def check_gradient(self, left_shape, right_shape):
        left = Tensor([(index % 7 - 3) / 5 for index in range(prod(left_shape))], shape=left_shape, requires_grad=True)
        right = Tensor([(index % 5 - 2) / 4 for index in range(prod(right_shape))], shape=right_shape, requires_grad=True)
        result = qt.matmul(left, right)
        coefficients = [(index % 11 - 5) / 7 for index in range(result.size)]
        result.backward(Tensor(coefficients, shape=result.shape))
        epsilon = 1e-6
        for parent in (left, right):
            for index, analytic in enumerate(parent.grad.values):
                original = parent.values[index]
                parent.values[index] = original + epsilon
                upper = sum(value * coefficient for value, coefficient in zip(qt.matmul(left, right).values, coefficients, strict=True))
                parent.values[index] = original - epsilon
                lower = sum(value * coefficient for value, coefficient in zip(qt.matmul(left, right).values, coefficients, strict=True))
                parent.values[index] = original
                self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=7)

    def test_vectors_and_matrices_return_expected_shapes_and_values(self):
        self.assertEqual(qt.matmul([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(qt.matmul([1, 2], [[3, 4, 5], [6, 7, 8]]), [15, 18, 21])
        self.assertEqual(qt.matmul([[1, 2], [3, 4]], [5, 6]), [17, 39])
        result = qt.matmul(Tensor([1, 2]), [3, 4])
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 11)

    def test_vectors_with_batched_matrices(self):
        matrices = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        self.assertEqual(qt.matmul([1, 2], matrices), [[7, 10], [19, 22]])
        self.assertEqual(qt.matmul(matrices, [1, 2]), [[5, 11], [17, 23]])

    def test_both_batch_axes_broadcast_without_mixing_matrices(self):
        left = Tensor(list(range(1, 13)), shape=(2, 1, 2, 3))
        right = Tensor(list(range(1, 19)), shape=(1, 3, 3, 2))
        a, b = left.tolist(), right.tolist()
        expected = [[[[sum(a[group][0][row][k] * b[0][batch][k][column] for k in range(3))
                       for column in range(2)] for row in range(2)]
                     for batch in range(3)] for group in range(2)]
        result = qt.matmul(left, right)
        self.assertEqual(result.shape, (2, 3, 2, 2))
        self.assertEqual(result.tolist(), expected)

    def test_shared_matrix_gradient_sums_all_batches(self):
        left = Tensor(list(range(1, 13)), shape=(2, 2, 3), requires_grad=True)
        weight = Tensor([[7, 8], [9, 10], [11, 12]], requires_grad=True)
        qt.sum(qt.matmul(left, weight)).backward()
        self.assertEqual(left.grad.tolist(), [[[15, 19, 23]] * 2] * 2)
        self.assertEqual(weight.grad.tolist(), [[22, 22], [26, 26], [30, 30]])

    def test_explicit_vector_gradient(self):
        vector = Tensor([1, 2], requires_grad=True)
        matrices = Tensor([[[3, 4], [5, 6]], [[7, 8], [9, 10]]], requires_grad=True)
        qt.matmul(vector, matrices).backward([[1, 2], [3, 4]])
        self.assertEqual(vector.grad.tolist(), [64, 84])
        self.assertEqual(matrices.grad.tolist(), [[[1, 2], [2, 4]], [[3, 4], [6, 8]]])

    def test_finite_differences_cover_vectors_batching_and_broadcast(self):
        for left_shape, right_shape in [
            ((3,), (3,)),
            ((3,), (2, 3, 2)),
            ((2, 2, 3), (3,)),
            ((2, 2, 3), (3, 2)),
            ((2, 3), (2, 3, 2)),
            ((2, 1, 2, 3), (1, 3, 3, 2)),
        ]:
            with self.subTest(left=left_shape, right=right_shape):
                self.check_gradient(left_shape, right_shape)

    def test_shared_batched_input_accumulates_both_paths(self):
        tensor = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        qt.sum(qt.matmul(tensor, tensor)).backward()
        self.assertEqual(tensor.grad.tolist(), [[[7, 11], [9, 13]], [[23, 27], [25, 29]]])

    def test_empty_axes_and_empty_contractions_have_zero_gradients(self):
        for left_shape, right_shape, output_shape in [
            ((0,), (0,), ()),
            ((0,), (0, 3), (3,)),
            ((2, 0), (0, 3), (2, 3)),
            ((0, 3), (3,), (0,)),
            ((2, 3), (3, 0), (2, 0)),
            ((0, 2, 3), (1, 3, 4), (0, 2, 4)),
            ((1, 2, 3), (0, 3, 4), (0, 2, 4)),
        ]:
            with self.subTest(left=left_shape, right=right_shape):
                left = Tensor([1.0] * prod(left_shape), shape=left_shape, requires_grad=True)
                right = Tensor([1.0] * prod(right_shape), shape=right_shape, requires_grad=True)
                result = qt.matmul(left, right)
                self.assertEqual(result.shape, output_shape)
                self.assertEqual(result.values, [0.0] * prod(output_shape))
                qt.sum(result).backward()
                self.assertEqual(left.grad.shape, left_shape)
                self.assertEqual(right.grad.shape, right_shape)
                self.assertEqual(left.grad.values, [0.0] * left.size)
                self.assertEqual(right.grad.values, [0.0] * right.size)

    def test_two_dimensional_accumulation_order_is_unchanged(self):
        self.assertEqual(cpu.matmul([1e16, 1, -1e16], [1, 1, 1], (1, 3), (3, 1)), [0.0])

    def test_scalar_inner_dimension_and_batch_mismatches_raise(self):
        for left, right in [
            (Tensor(2), Tensor([1, 2])),
            (Tensor([1, 2]), Tensor(2)),
            (Tensor([1, 2]), Tensor([1, 2, 3])),
            (Tensor([0.0] * 12, shape=(2, 2, 3)), Tensor([0.0] * 18, shape=(3, 3, 2))),
        ]:
            with self.subTest(left=left.shape, right=right.shape), self.assertRaises(ValueError):
                qt.matmul(left, right)


if __name__ == "__main__":
    unittest.main()
