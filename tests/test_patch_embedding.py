import random
import unittest
from math import prod

from qutai import Tensor, nn


class PatchEmbeddingTests(unittest.TestCase):
    def test_patch_order_projection_and_hand_computed_backward(self):
        layer = nn.PatchEmbedding(1, 2, 2, seed=42)
        layer.projection.weight.values = [1.0] * 4 + [1.0, 0.0, 0.0, -1.0]
        layer.projection.bias.values = [0.5, -0.5]
        inputs = Tensor(list(range(1, 17)), shape=(1, 1, 4, 4), requires_grad=True)
        output = layer(inputs)
        self.assertEqual(output.tolist(), [[[14.5, -5.5], [22.5, -5.5], [46.5, -5.5], [54.5, -5.5]]])
        output.sum().backward()
        self.assertEqual(inputs.grad.tolist(), [[[[2, 1, 2, 1], [1, 0, 1, 0], [2, 1, 2, 1], [1, 0, 1, 0]]]])
        self.assertEqual(layer.projection.weight.grad.values, [24, 28, 40, 44] * 2)
        self.assertEqual(layer.projection.bias.grad.tolist(), [4, 4])

    def test_rectangular_patches_multiple_channels_and_empty_batch(self):
        layer = nn.PatchEmbedding(2, 3, (2, 3), seed=42)
        inputs = Tensor([index / 10 for index in range(96)], shape=(2, 2, 4, 6), requires_grad=True)
        output = layer(inputs)
        self.assertEqual(output.shape, (2, 4, 3))
        output.square().sum().backward()
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        empty = Tensor([], shape=(0, 2, 4, 6), requires_grad=True)
        layer(empty).sum().backward()
        self.assertEqual(layer(empty).shape, (0, 4, 3))
        self.assertEqual(empty.grad.values, [])

    def test_patch_dimensions_do_not_silently_discard_border_pixels(self):
        layer = nn.PatchEmbedding(1, 2, 2)
        with self.assertRaisesRegex(ValueError, '整除'):
            layer(Tensor([0.0] * 12, shape=(1, 1, 3, 4)))
        with self.assertRaisesRegex(ValueError, '四维'):
            layer([[1, 2]])
        with self.assertRaises(ValueError):
            nn.PatchEmbedding(1, 2, 0)

    def test_position_rows_broadcast_and_accumulate_only_used_positions(self):
        layer = nn.PositionEmbedding(4, 3, seed=42)
        layer.weight.values = [float(index) for index in range(12)]
        inputs = Tensor([0.0] * 12, shape=(2, 2, 3), requires_grad=True)
        output = layer(inputs)
        self.assertEqual(output.tolist(), [[[0, 1, 2], [3, 4, 5]]] * 2)
        output.sum().backward()
        self.assertEqual(inputs.grad.values, [1] * 12)
        self.assertEqual(layer.weight.grad.tolist(), [[2, 2, 2], [2, 2, 2], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(layer(Tensor([], shape=(2, 0, 3))).shape, (2, 0, 3))

    def test_position_initialization_and_invalid_sequence_shapes(self):
        state = random.getstate()
        first = nn.PositionEmbedding(4, 3, seed=42)
        second = nn.PositionEmbedding(4, 3, seed=42)
        self.assertEqual(first.weight.values, second.weight.values)
        self.assertEqual(random.getstate(), state)
        for shape in ((1, 5, 3), (1, 2, 4), (2, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                first(Tensor([0] * prod(shape), shape=shape))
        for args in ((0, 2), (2, 0)):
            with self.assertRaises(ValueError):
                nn.PositionEmbedding(*args)

    def test_mean_layer_keeps_tensor_and_backpropagates(self):
        inputs = Tensor([[[1, 2], [3, 4]]], requires_grad=True)
        output = nn.Mean(axis=1)(inputs)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.tolist(), [[2, 3]])
        output.sum().backward()
        self.assertEqual(inputs.grad.values, [0.5] * 4)
        self.assertEqual(nn.Mean(axis=None, keepdims=True)(inputs).shape, (1, 1, 1))
        self.assertEqual(nn.Mean().parameters(), ())


if __name__ == '__main__':
    unittest.main()
