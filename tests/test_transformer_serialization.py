import json
import unittest
from math import exp, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, nn


def image_model(seed):
    return nn.Model([
        nn.PatchEmbedding(1, 4, (2, 3), seed=seed),
        nn.PositionEmbedding(6, 4, seed=seed + 1),
        nn.TransformerBlock(4, 2, hidden_features=6, causal=True, eps=0.02, seed=seed + 2),
        nn.Mean(axis=1), nn.Linear(4, 2, seed=seed + 10),
    ])


class TransformerSerializationTests(unittest.TestCase):
    def test_norm_gelu_feed_forward_and_mean_roundtrip(self):
        inputs = Tensor([(index - 10) / 10 for index in range(24)], shape=(2, 3, 4))
        for axis in (1, None):
            with self.subTest(axis=axis), TemporaryDirectory() as folder:
                source = nn.Model([nn.LayerNorm(4, eps=0.03), nn.GELU(), nn.FeedForward(4, seed=42), nn.Mean(axis=axis, keepdims=True)])
                path = Path(folder) / 'layers.json'
                source.save(path)
                restored = nn.Model().load(path)
                self.assertEqual([type(layer) for layer in restored.layers], [nn.LayerNorm, nn.GELU, nn.FeedForward, nn.Mean])
                self.assertEqual(restored.layers[0].eps, 0.03)
                self.assertEqual(restored.layers[2].layers[0].weight.shape, (4, 16))
                self.assertEqual((restored.layers[3].axis, restored.layers[3].keepdims), (axis, True))
                self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)

    def test_multi_head_attention_preserves_heads_mask_and_output_projection(self):
        inputs = [[[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3], [-0.1, 0.7, 0.2, 0.4]]]
        for heads, causal in ((1, False), (2, True)):
            with self.subTest(heads=heads, causal=causal), TemporaryDirectory() as folder:
                source = nn.Model([nn.MultiHeadAttention(4, heads, causal=causal, seed=42)])
                path = Path(folder) / 'attention.json'
                source.save(path)
                restored = nn.Model().load(path)
                layer = restored.layers[0]
                self.assertIs(type(layer), nn.MultiHeadAttention)
                self.assertEqual((layer.heads, layer.causal), (heads, causal))
                self.assertEqual(layer.weights(inputs).shape, (1, heads, 3, 3))
                self.assertEqual(layer.weights(inputs).values, source.layers[0].weights(inputs).values)
                self.assertEqual(layer.output.weight.values, source.layers[0].output.weight.values)
                self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)

    def test_image_transformer_rebuilds_and_can_continue_training(self):
        source = image_model(42)
        inputs = Tensor([(index - 20) / 20 for index in range(48)], shape=(2, 1, 4, 6))
        prediction = source(inputs, train=False).values
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'image.json'
            source.save(path)
            self.assertEqual(set(json.loads(path.read_text())), {'model', 'parameters'})
            restored = nn.Model().load(path)
        self.assertIsInstance(restored.layers[0], nn.PatchEmbedding)
        self.assertEqual(restored.layers[0].patch_size, (2, 3))
        self.assertIsInstance(restored.layers[1], nn.PositionEmbedding)
        self.assertEqual(restored.layers[1].weight.shape, (6, 4))
        block = restored.layers[2]
        self.assertIsInstance(block, nn.TransformerBlock)
        self.assertEqual((block.attention.heads, block.attention.causal), (2, True))
        self.assertEqual(block.feed_forward.layers[0].weight.shape, (4, 6))
        self.assertEqual((block.attention_norm.eps, block.feed_forward_norm.eps), (0.02, 0.02))
        result = restored(inputs, train=False)
        self.assertEqual(result.values, prediction)
        self.assertFalse(result.requires_grad)
        before = [parameter.values[:] for parameter in restored.parameters()]
        optimizer = nn.SGD(restored, learning_rate=0.01)
        restored(inputs).square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in restored.parameters()))
        optimizer.step()
        self.assertTrue(any(parameter.values != values for parameter, values in zip(restored.parameters(), before, strict=True)))

    def test_existing_load_keeps_all_parameter_and_child_references(self):
        source, restored = image_model(42), image_model(7)
        inputs = Tensor([(index - 20) / 20 for index in range(48)], shape=(2, 1, 4, 6))
        restored.parameters()[0].requires_grad = False
        optimizer = nn.SGD(restored, learning_rate=0.01)
        flags = [parameter.requires_grad for parameter in restored.parameters()]
        block = restored.layers[2]
        children = (block.attention_norm, block.attention, block.feed_forward_norm, block.feed_forward)
        restored(inputs).square().mean().backward()
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'image.json'
            source.save(path)
            self.assertIs(restored.load(path), restored)
        for previous, current in zip(optimizer.parameters, restored.parameters(), strict=True):
            self.assertIs(previous, current)
            self.assertIsNone(current.grad)
        self.assertEqual([parameter.requires_grad for parameter in restored.parameters()], flags)
        current_block = restored.layers[2]
        for previous, current in zip(children, (current_block.attention_norm, current_block.attention, current_block.feed_forward_norm, current_block.feed_forward), strict=True):
            self.assertIs(previous, current)
        self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)
        restored(inputs).square().mean().backward()
        optimizer.step()
        self.assertIsNone(restored.parameters()[0].grad)
        self.assertIsNotNone(restored.parameters()[-1].grad)

    def test_shared_embeddings_norms_and_projections_stay_shared(self):
        positions = nn.PositionEmbedding(6, 4, seed=42)
        positions.weight.requires_grad = False
        block = nn.TransformerBlock(4, 2, hidden_features=6, causal=True, seed=43)
        block.attention.key.weight = block.attention.query.weight
        block.feed_forward_norm.weight = block.attention_norm.weight
        source = nn.Model([positions, positions, block, block])
        inputs = Tensor([(index - 5) / 10 for index in range(12)], shape=(1, 3, 4))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'shared.json'
            source.save(path)
            self.assertEqual(len(json.loads(path.read_text())['parameters']), len(source.parameters()))
            restored = nn.Model().load(path)
        self.assertIs(restored.layers[0].weight, restored.layers[1].weight)
        self.assertFalse(restored.layers[0].weight.requires_grad)
        first, second = restored.layers[2:]
        self.assertIs(first.attention.query.weight, first.attention.key.weight)
        self.assertIs(first.attention_norm.weight, first.feed_forward_norm.weight)
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            self.assertIs(left, right)
        self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)

    def test_different_head_or_mask_configuration_is_rejected_before_loading(self):
        source = nn.Model([nn.MultiHeadAttention(4, 2, causal=True, seed=42)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'attention.json'
            source.save(path)
            for heads, causal in ((1, True), (2, False)):
                with self.subTest(heads=heads, causal=causal):
                    target = nn.Model([nn.MultiHeadAttention(4, heads, causal=causal, seed=7)])
                    before = [parameter.values[:] for parameter in target.parameters()]
                    with self.assertRaisesRegex(ValueError, '结构'):
                        target.load(path)
                    self.assertEqual([parameter.values for parameter in target.parameters()], before)

    def test_original_attention_archive_loads_without_becoming_multi_head(self):
        records = [{'shape': [2, 2], 'values': [1, 0, 0, 1]}, {'shape': [2], 'values': [0, 0]}] * 3
        attention = {'type': 'Attention', **{name: {'type': 'Linear', 'in_features': 2, 'out_features': 2, 'parameters': [2 * index, 2 * index + 1]}
                     for index, name in enumerate(('query', 'key', 'value'))}}
        archive = {'model': {'type': 'Model', 'layers': [attention]}, 'parameters': records}
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps(archive), encoding='utf-8')
            restored = nn.Model().load(path)
            restored.save(path)
            self.assertEqual(set(json.loads(path.read_text())['model']['layers'][0]), {'type', 'query', 'key', 'value'})
        self.assertIs(type(restored.layers[0]), nn.Attention)
        inputs = [[[1, 0], [0, 1]]]
        self.assertEqual(restored.layers[0].weights(inputs).shape, (1, 2, 2))
        probability = exp(1 / sqrt(2)) / (exp(1 / sqrt(2)) + 1)
        for value, expected in zip(restored(inputs, train=False).values, (probability, 1 - probability, 1 - probability, probability), strict=True):
            self.assertAlmostEqual(value, expected)


if __name__ == '__main__':
    unittest.main()
