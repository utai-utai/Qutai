import math
import unittest

from qutai import Tensor, nn
from qutai.core.tensor import gelu


class TransformerTests(unittest.TestCase):
    def make_identity_attention(self, heads=2, causal=False):
        layer = nn.MultiHeadAttention(2, heads, causal=causal, seed=7)
        for projection in (layer.query, layer.key, layer.value, layer.output):
            projection.weight.values = [1.0, 0.0, 0.0, 1.0]
            projection.bias.values = [0.0, 0.0]
        return layer

    def check_gradients(self, layer, inputs):
        result = layer(inputs)
        seed = [(index % 5 - 2) / 3 for index in range(result.size)]
        result.backward(Tensor(seed, shape=result.shape))
        epsilon = 1e-6
        for parent in (inputs, *layer.parameters()):
            self.assertIsNotNone(parent.grad)
            for index, analytic in enumerate(parent.grad.values):
                with self.subTest(layer=type(layer).__name__, shape=parent.shape, index=index):
                    original = parent.values[index]
                    parent.values[index] = original + epsilon
                    upper = layer(inputs)
                    parent.values[index] = original - epsilon
                    lower = layer(inputs)
                    parent.values[index] = original
                    numeric = math.fsum((a - b) * incoming for a, b, incoming in zip(upper.values, lower.values, seed, strict=True)) / (2 * epsilon)
                    self.assertAlmostEqual(analytic, numeric, places=6)

    def test_two_heads_identity_projections_match_hand_calculation(self):
        layer = self.make_identity_attention()
        inputs = [[[1, 0], [0, 1]]]
        probability = math.exp(1) / (math.exp(1) + 1)
        weights = layer.weights(inputs)
        self.assertEqual(weights.shape, (1, 2, 2, 2))
        expected_weights = [probability, 1 - probability, 0.5, 0.5, 0.5, 0.5, 1 - probability, probability]
        for actual, expected in zip(weights.values, expected_weights, strict=True):
            self.assertAlmostEqual(actual, expected)
        result = layer(inputs)
        self.assertEqual(result.shape, (1, 2, 2))
        for actual, expected in zip(result.values, [probability, 0.5, 0.5, probability], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_multihead_inputs_and_all_qkvo_gradients_match_central_differences(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                layer = nn.MultiHeadAttention(2, 2, causal=causal, seed=7)
                inputs = Tensor([[[0.2, -0.4], [0.7, 0.1]]], requires_grad=True)
                self.check_gradients(layer, inputs)
                for projection in (layer.query, layer.key, layer.value, layer.output):
                    self.assertIsNotNone(projection.weight.grad)
                    self.assertIsNotNone(projection.bias.grad)

    def test_causal_weights_are_zero_above_diagonal(self):
        layer = self.make_identity_attention(causal=True)
        weights = layer.weights([[[1, 0], [0, 1], [2, 3]]])
        for head in weights.tolist()[0]:
            for row, probabilities in enumerate(head):
                self.assertAlmostEqual(sum(probabilities), 1)
                self.assertEqual(probabilities[row + 1:], [0.0] * (2 - row))
                self.assertTrue(all(value >= 0 for value in probabilities))
            self.assertEqual(head[0], [1, 0, 0])

    def test_causal_attention_and_block_cannot_read_future_inputs(self):
        for layer in (self.make_identity_attention(causal=True),
                      nn.TransformerBlock(2, 2, hidden_features=3, causal=True, eps=0.2, seed=7)):
            with self.subTest(layer=type(layer).__name__):
                inputs = Tensor([[[0.2, -0.4], [0.7, 0.1], [0.3, 0.8]]], requires_grad=True)
                changed = Tensor([[[0.2, -0.4], [0.7, 0.1], [100, -200]]])
                result = layer(inputs)
                self.assertEqual(result[:, :2].tolist(), layer(changed)[:, :2].tolist())
                result[:, :2].sum().backward()
                self.assertEqual(inputs.grad.values[-2:], [0.0, 0.0])

    def test_noncausal_attention_keeps_access_to_future_inputs(self):
        layer = self.make_identity_attention(causal=False)
        first = layer([[[1, 0], [0, 1], [2, 3]]])
        changed = layer([[[1, 0], [0, 1], [100, -200]]])
        self.assertNotEqual(first[:, :2].tolist(), changed[:, :2].tolist())

    def test_original_attention_semantics_are_preserved(self):
        original = nn.Attention(2, seed=7)
        multihead = self.make_identity_attention(heads=1)
        for projection in (original.query, original.key, original.value):
            projection.weight.values = [1.0, 0.0, 0.0, 1.0]
            projection.bias.values = [0.0, 0.0]
        inputs = [[[1, 0], [0, 1]]]
        probability = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        self.assertEqual(original.weights(inputs).shape, (1, 2, 2))
        self.assertEqual(len(original.parameters()), 6)
        for actual, expected in zip(original(inputs).values, [probability, 1 - probability, 1 - probability, probability], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(multihead(inputs).tolist(), original(inputs).tolist())
        self.assertEqual(multihead.weights(inputs)[:, 0].tolist(), original.weights(inputs).tolist())
        self.assertEqual(nn.Attention(2, 3, seed=7)(inputs).shape, (1, 2, 3))

    def test_feed_forward_matches_linear_gelu_linear_at_each_position(self):
        layer = nn.FeedForward(2, 2, seed=7)
        layer.layers[0].weight.values = [1.0, 0.0, 0.0, 1.0]
        layer.layers[0].bias.values = [0.0, 0.0]
        layer.layers[2].weight.values = [1.0, 2.0, 3.0, 4.0]
        layer.layers[2].bias.values = [0.1, 0.2]
        inputs = [[[0.5, -0.2], [1.0, 0.3]], [[-0.1, 0.7], [0.2, -0.8]]]
        result = layer(inputs)
        self.assertEqual(result.shape, (2, 2, 2))
        for batch, expected_batch in zip(result.tolist(), inputs, strict=True):
            for row, (a, b) in zip(batch, expected_batch, strict=True):
                a, b = a * (1 + math.erf(a / math.sqrt(2))) / 2, b * (1 + math.erf(b / math.sqrt(2))) / 2
                self.assertAlmostEqual(row[0], a + 3 * b + 0.1)
                self.assertAlmostEqual(row[1], 2 * a + 4 * b + 0.2)

    def test_transformer_all_parameters_and_inputs_match_central_differences(self):
        block = nn.TransformerBlock(2, 2, hidden_features=3, causal=True, eps=0.2, seed=7)
        inputs = Tensor([[[0.2, -0.4], [0.7, 0.1]]], requires_grad=True)
        self.check_gradients(block, inputs)
        self.assertEqual(len(block.parameters()), 16)

    def test_eval_has_same_values_without_a_graph(self):
        model = nn.Model([nn.TransformerBlock(2, 2, hidden_features=3, seed=7)])
        inputs = Tensor([[[0.2, -0.4], [0.7, 0.1]]], requires_grad=True)
        training, prediction = model(inputs), model(inputs, train=False)
        self.assertEqual(prediction.tolist(), training.tolist())
        self.assertTrue(training.requires_grad)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertTrue(inputs.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_empty_batch_keeps_shape_and_zero_parameter_gradients(self):
        for layer in (nn.MultiHeadAttention(2, 2, seed=7), nn.TransformerBlock(2, 2, hidden_features=3, seed=7)):
            with self.subTest(layer=type(layer).__name__):
                inputs = Tensor([], shape=(0, 2, 2), requires_grad=True)
                result = layer(inputs)
                self.assertEqual(result.shape, inputs.shape)
                result.sum().backward()
                self.assertEqual(inputs.grad.values, [])
                for parameter in layer.parameters():
                    self.assertEqual(parameter.grad.values, [0.0] * parameter.size)

    def test_gelu_matches_exact_erf_formula_and_central_differences(self):
        inputs = Tensor([-5, -1, 0, 0.3, 2], requires_grad=True)
        layer = nn.GELU()
        result = layer(inputs)
        seed = [0.2, -0.7, 1.5, 0.3, -0.6]
        result.backward(seed)
        for index, value in enumerate(inputs.values):
            self.assertAlmostEqual(result.values[index], value * (1 + math.erf(value / math.sqrt(2))) / 2)
            upper, lower = inputs.values[:], inputs.values[:]
            upper[index] += 1e-6
            lower[index] -= 1e-6
            numeric = (gelu(upper)[index] - gelu(lower)[index]) * seed[index] / 2e-6
            self.assertAlmostEqual(inputs.grad.values[index], numeric, places=7)
        self.assertEqual(layer.parameters(), ())
        self.assertEqual(gelu([0.0]), [0.0])

    def test_invalid_features_heads_shapes_and_norm_configuration(self):
        for features, heads in ((0, 1), (-2, 1), (2, 0), (2, -1), (3, 2), (2, 3)):
            with self.subTest(features=features, heads=heads), self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                nn.MultiHeadAttention(features, heads)
        for features, eps in ((0, 1e-5), (2, 0), (2, -1)):
            with self.subTest(features=features, eps=eps), self.assertRaises(ValueError):
                nn.LayerNorm(features, eps=eps)
        layer = nn.MultiHeadAttention(2, 2, seed=7)
        with self.assertRaisesRegex(ValueError, "三维"):
            layer([[1, 2]])
        with self.assertRaisesRegex(ValueError, "矩阵"):
            layer([[[1, 2, 3]]])
        with self.assertRaises(ValueError):
            nn.FeedForward(2, 0)


if __name__ == "__main__":
    unittest.main()
