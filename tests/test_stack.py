"""新增轴的组合、输出类别、共享输入和高维反向传播。"""

import unittest

import qutai as qt
from qutai import Tensor
from qutai.core.context import set_recording


class StackTests(unittest.TestCase):
    def test_scalars_and_plain_vectors_follow_input_return_rules(self):
        self.assertEqual(qt.stack([1, 2, 3]), [1.0, 2.0, 3.0])
        self.assertEqual(qt.stack([1, 2, 3], axis=-1), [1.0, 2.0, 3.0])
        self.assertEqual(qt.stack([[1, 2], [3, 4]]), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(qt.stack([[1, 2], [3, 4]], axis=1), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(qt.stack([[1, 2], [3, 4]], axis=-1), [[1.0, 3.0], [2.0, 4.0]])

    def test_matrix_middle_axis_preserves_element_order(self):
        left = [[1, 2, 3], [4, 5, 6]]
        right = [[7, 8, 9], [10, 11, 12]]
        self.assertEqual(qt.stack([left, right], axis=1),
                         [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]])
        self.assertEqual(qt.stack([left, right], axis=-1),
                         [[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]])

    def test_generator_and_mixed_data_keep_tensor_graph(self):
        leaf = Tensor([1, 2], requires_grad=True)
        result = qt.stack(item for item in ([3, 4], leaf))
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (2, 2))
        result.backward([[5, 6], [7, 8]])
        self.assertEqual(leaf.grad.tolist(), [7, 8])
        self.assertEqual(qt.stack([Tensor(3), 4]).tolist(), [3, 4])

    def test_shared_tensor_contributions_accumulate(self):
        leaf = Tensor([0.3, -0.8], requires_grad=True)
        result = qt.stack([leaf, leaf, leaf], axis=-1)
        result.backward([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(leaf.grad.tolist(), [6, 15])
        self.assertEqual(result.parents, (leaf, leaf, leaf))

    def test_multidimensional_axes_and_gradients_match_finite_differences(self):
        data = [[[0.2, -0.4, 0.7]], [[-0.3, 0.8, 0.1]]]
        epsilon = 1e-6
        for axis in (0, 1, 2, 3, -1):
            with self.subTest(axis=axis):
                parents = [Tensor(data, requires_grad=True), Tensor(data, requires_grad=True)]
                result = qt.stack(parents, axis=axis)
                coefficients = Tensor([(i % 7 - 3) / 5 for i in range(result.size)], shape=result.shape)
                (result.square() * coefficients).sum().backward()
                for parent in parents:
                    for index, analytic in enumerate(parent.grad.values):
                        original = parent.values[index]
                        parent.values[index] = original + epsilon
                        upper = (qt.stack(parents, axis=axis).square() * coefficients).sum().item()
                        parent.values[index] = original - epsilon
                        lower = (qt.stack(parents, axis=axis).square() * coefficients).sum().item()
                        parent.values[index] = original
                        self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)

    def test_empty_dimensions_preserve_shapes_and_zero_gradients(self):
        for axis, expected in ((0, (2, 2, 0)), (1, (2, 2, 0)), (2, (2, 0, 2))):
            with self.subTest(axis=axis):
                left = Tensor([], shape=(2, 0), requires_grad=True)
                right = Tensor([], shape=(2, 0), requires_grad=True)
                result = qt.stack([left, right], axis=axis)
                self.assertEqual(result.shape, expected)
                result.sum().backward()
                for parent in (left, right):
                    self.assertEqual(parent.grad.shape, (2, 0))
                    self.assertEqual(parent.grad.values, [])

    def test_recording_context_detaches_without_changing_inputs(self):
        leaf = Tensor(0.3, requires_grad=True)
        with set_recording(False):
            result = qt.stack([leaf, leaf])
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())
        self.assertTrue(leaf.requires_grad)
        self.assertIsNone(leaf.grad)

    def test_requires_nonempty_equal_shapes_and_a_valid_new_axis(self):
        for items, axis in (([], 0), ([1, [2]], 0), ([[1, 2], [3]], 0), ([1, 2], 1),
                            ([[1, 2], [3, 4]], 2), ([[1, 2], [3, 4]], -3)):
            with self.subTest(items=items, axis=axis), self.assertRaises(ValueError):
                qt.stack(items, axis=axis)


if __name__ == '__main__':
    unittest.main()
