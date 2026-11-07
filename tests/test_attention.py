import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, nn


class AttentionTests(unittest.TestCase):
    def test_identity_projections_match_hand_computed_attention(self):
        layer = nn.Attention(2, seed=42)
        for projection in (layer.query, layer.key, layer.value):
            projection.weight.values = [1.0, 0.0, 0.0, 1.0]
            projection.bias.values = [0.0, 0.0]
        inputs = Tensor([[[1, 0], [0, 1]]])
        probability = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        result = layer(inputs)
        expected = [probability, 1 - probability, 1 - probability, probability]
        self.assertEqual(result.shape, (1, 2, 2))
        for actual, target in zip(result.values, expected, strict=True):
            self.assertAlmostEqual(actual, target)
        self.assertEqual(layer.weights(inputs).values, result.values)

    def test_all_input_and_projection_gradients_match_finite_differences(self):
        layer = nn.Attention(2, 3, seed=7)
        inputs = Tensor([[[0.2, -0.4], [0.7, 0.1]], [[-0.3, 0.5], [0.6, -0.8]]], requires_grad=True)
        layer(inputs).square().mean().backward()
        epsilon = 1e-6
        for tensor in (inputs, *layer.parameters()):
            self.assertIsNotNone(tensor.grad)
            for index, expected in enumerate(tensor.grad.values):
                with self.subTest(shape=tensor.shape, index=index):
                    original = tensor.values[index]
                    tensor.values[index] = original + epsilon
                    upper = layer(inputs).square().mean().item()
                    tensor.values[index] = original - epsilon
                    lower = layer(inputs).square().mean().item()
                    tensor.values[index] = original
                    self.assertAlmostEqual(expected, (upper - lower) / (2 * epsilon), places=6)

    def test_rows_are_probabilities_and_eval_pauses_recording(self):
        layer = nn.Attention(2, 4, seed=42)
        inputs = Tensor([[[1, 0], [0, 1], [2, 1]]], requires_grad=True)
        weights = layer.weights(inputs)
        self.assertEqual(weights.shape, (1, 3, 3))
        for row in weights.tolist()[0]:
            self.assertAlmostEqual(sum(row), 1)
            self.assertTrue(all(0 <= value <= 1 for value in row))
        model = nn.Model([layer, nn.Linear(4, 1, seed=43)])
        result = model(inputs, train=False)
        self.assertEqual(result.shape, (1, 3, 1))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())
        self.assertTrue(model(inputs).requires_grad)
        with self.assertRaisesRegex(ValueError, "三维"):
            layer([[1, 2]])

    def test_save_load_and_in_place_loading_preserve_projections(self):
        model = nn.Model([nn.Attention(2, 3, seed=42), nn.Linear(3, 1, seed=43)])
        inputs = [[[0.1, 1], [0.2, 0], [0.4, 0]]]
        optimizer = nn.SGD(model, learning_rate=0.1)
        original_parameters = model.parameters()
        model(inputs).square().mean().backward()
        optimizer.step()
        scores = model(inputs, train=False).tolist()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'attention.json'
            model.save(path)
            restored = nn.Model().load(path)
            self.assertIsInstance(restored.layers[0], nn.Attention)
            self.assertEqual(restored(inputs, train=False).tolist(), scores)
            self.assertEqual(restored.layers[0].weights(inputs).tolist(), model.layers[0].weights(inputs).tolist())
            model.parameters()[0].values[0] += 1
            self.assertIs(model.load(path), model)
            self.assertEqual(model.parameters(), original_parameters)
            self.assertEqual(optimizer.parameters, original_parameters)
            self.assertEqual(model(inputs, train=False).tolist(), scores)
            self.assertTrue(all(parameter.grad is None for parameter in original_parameters))

    def test_sequence_linear_accumulates_shared_weight_gradients(self):
        layer = nn.Linear(2, 1, seed=42)
        inputs = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        layer.weight.values = [2.0, 3.0]
        output = layer(inputs)
        self.assertEqual(output.shape, (2, 2, 1))
        self.assertEqual(output.tolist(), [[[8.0], [18.0]], [[28.0], [38.0]]])
        output.sum().backward()
        self.assertEqual(layer.weight.grad.tolist(), [[16.0], [20.0]])
        self.assertEqual(layer.bias.grad.tolist(), [4.0])
        self.assertEqual(inputs.grad.values, [2.0, 3.0] * 4)


if __name__ == '__main__':
    unittest.main()
