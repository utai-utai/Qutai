import unittest
from math import log

from qutai import Tensor, nn


class SequenceCrossEntropyTests(unittest.TestCase):
    def setUp(self):
        self.loss = nn.CrossEntropy()

    def test_three_and_four_dimensions_match_flattened_values_and_gradients(self):
        values = [(index % 9 - 4) / 5 for index in range(24)]
        cases = (
            ((2, 3, 4), ((0, 2, 1), (3, 0, 2))),
            ((2, 1, 3, 4), (((0, 2, 1),), ((3, 0, 2),))),
        )
        for shape, labels in cases:
            with self.subTest(shape=shape):
                scores = Tensor(values, shape=shape, requires_grad=True)
                flat = Tensor(values, shape=(6, 4), requires_grad=True)
                actual, expected = self.loss(scores, labels), self.loss(flat, (0, 2, 1, 3, 0, 2))
                self.assertEqual(actual.item(), expected.item())
                actual.backward(2.3)
                expected.backward(2.3)
                self.assertEqual(scores.grad.shape, shape)
                self.assertEqual(scores.grad.values, flat.grad.values)

    def test_plain_nested_tuples_return_scalar_tensor(self):
        loss = self.loss((((0, 0, 0), (0, 0, 0)),), ((1, 2),))
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(loss.item(), log(3))
        self.assertFalse(loss.requires_grad)

    def test_scaled_gradient_matches_central_finite_differences(self):
        values = [0.2, -0.4, 0.7, 0.3, 0.1, -0.6, 0.8, -0.2, 0.5, 0.9, -0.3, 0.4]
        labels = ((2, 0), (1, 2))
        scores = Tensor(values, shape=(2, 2, 3), requires_grad=True)
        scale = -1.7
        self.loss(scores, labels).backward(scale)
        for index, analytic in enumerate(scores.grad.values):
            with self.subTest(index=index):
                high, low = Tensor(values, shape=scores.shape), Tensor(values, shape=scores.shape)
                high.values[index] += 1e-6
                low.values[index] -= 1e-6
                numeric = scale * (self.loss(high, labels).item() - self.loss(low, labels).item()) / 2e-6
                self.assertAlmostEqual(analytic, numeric, places=7)

    def test_empty_axes_invalid_shapes_and_labels_have_chinese_errors(self):
        cases = (
            (Tensor([], shape=(0, 2, 3)), []),
            (Tensor([], shape=(2, 0, 3)), [[], []]),
            (Tensor([], shape=(2, 2, 0)), [[0, 0], [0, 0]]),
            ([1, 2], [0]),
            ([[[1, 2], [3, 4]]], [0, 1]),
            ([[[1, 2], [3, 4]]], [[0]]),
            ([[[1, 2], [3, 4]]], [[0, -1]]),
            ([[[1, 2], [3, 4]]], [[0, 2]]),
            ([[[1, 2], [3, 4]]], [[0, 0.5]]),
            ([[[1, 2], [3, 4]]], [[0, True]]),
        )
        for scores, labels in cases:
            with self.subTest(scores=scores, labels=labels):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    self.loss(scores, labels)

    def test_one_category_has_zero_loss_and_gradient(self):
        scores = Tensor([1000, -1000, 0, 42], shape=(2, 2, 1), requires_grad=True)
        loss = self.loss(scores, ((0, 0), (0, 0)))
        self.assertEqual(loss.item(), 0)
        loss.backward(3.4)
        self.assertEqual(scores.grad.shape, scores.shape)
        self.assertEqual(scores.grad.values, [0, 0, 0, 0])

    def test_prediction_loss_does_not_record_a_graph(self):
        model = nn.Model([nn.Linear(2, 3, seed=42)])
        scores = model([[[1, 0], [0, 1]], [[2, 1], [-1, 2]]], train=False)
        loss = self.loss(scores, ((0, 2), (1, 0)))
        self.assertFalse(loss.requires_grad)
        self.assertEqual(loss.parents, ())
        self.assertIsNone(loss.backward_rule)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
