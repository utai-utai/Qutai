"""增量续写与完整前向的一致性、计算量及缓存生命周期。"""

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qutai import lm, nn
from qutai.core.context import is_recording, is_training, set_recording, set_training
from qutai.nn.cache import KVCache


class ProjectionCounter:
    """只统计指定投影层实际接收的序列长度，仍执行真实计算。"""
    def __init__(self, layer):
        self.layer = layer
        self.original = nn.Linear.__call__
        self.lengths = []

    def __call__(self, layer, data):
        if layer is self.layer: self.lengths.append(data.shape[1])
        return self.original(layer, data)


class LanguageCacheTests(unittest.TestCase):
    def make_model(self, context=8, depth=2, seed=7):
        return lm.Model(lm.Tokenizer("abc"), context_length=context, features=4, heads=2,
                        hidden_features=6, depth=depth, seed=seed)

    def assert_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        for left, right in zip(actual.values, expected.values, strict=True):
            self.assertAlmostEqual(left, right, places=11)

    def test_prefill_and_incremental_chunks_match_full_prefix_logits(self):
        model = self.make_model(context=6)
        tokens = [[0, 1, 2, 0, 2], [2, 0, 1, 2, 1]]
        cache = {index: KVCache() for index, layer in enumerate(model.network.layers) if isinstance(layer, nn.TransformerBlock)}
        offset = 0
        for length in (2, 2, 1):
            current = [row[offset:offset + length] for row in tokens]
            actual = model._cached(current, cache, offset=offset)
            expected = model([row[:offset + length] for row in tokens], train=False)[:, offset:]
            self.assert_close(actual, expected)
            self.assertFalse(actual.requires_grad)
            self.assertEqual(actual.parents, ())
            offset += length
            for state in cache.values():
                self.assertEqual(state.length, offset)
                self.assertEqual(state.keys.shape, (2, 2, offset, 2))
                self.assertEqual(state.values.shape, (2, 2, offset, 2))
                self.assertFalse(state.keys.requires_grad)
                self.assertFalse(state.values.requires_grad)

    def test_reusing_one_block_keeps_a_separate_cache_for_each_network_position(self):
        model = self.make_model()
        layers = model.network.layers
        model.network.layers = (layers[0], layers[1], layers[1], *layers[-2:])
        cache = {index: KVCache() for index, layer in enumerate(model.network.layers) if isinstance(layer, nn.TransformerBlock)}
        self.assertIsNot(cache[1], cache[2])
        model._cached([[0, 1]], cache, offset=0)
        actual = model._cached([[2]], cache, offset=2)
        self.assert_close(actual, model([[0, 1, 2]], train=False)[:, -1:])
        self.assertEqual((cache[1].length, cache[2].length), (3, 3))

    def test_greedy_and_seeded_sampling_match_across_context_boundaries(self):
        for context, prompt, length in ((8, "a", 4), (8, "ab", 5), (3, "abc", 5),
                                        (3, "abcabc", 5), (1, "ab", 3)):
            model = self.make_model(context=context)
            for temperature in (0, 0.8):
                settings = dict(length=length, temperature=temperature, top_k=2, seed=42)
                with self.subTest(context=context, prompt=prompt, temperature=temperature):
                    expected = model.generate(prompt, use_cache=False, **settings)
                    self.assertEqual(model.generate(prompt, use_cache=True, **settings), expected)
                    self.assertEqual(model.generate(prompt, **settings), expected)
                    self.assertTrue(expected.startswith(prompt))
                    self.assertEqual(len(expected), len(prompt) + length)

    def test_cache_reuses_past_key_projections_until_the_window_is_full(self):
        model = self.make_model()
        projection = model.network.layers[1].attention.key
        cached, full = ProjectionCounter(projection), ProjectionCounter(projection)
        with patch.object(nn.Linear, "__call__", autospec=True, side_effect=cached):
            expected = model.generate("ab", length=5)
        with patch.object(nn.Linear, "__call__", autospec=True, side_effect=full):
            self.assertEqual(model.generate("ab", length=5, use_cache=False), expected)
        self.assertEqual(cached.lengths, [2, 1, 1, 1, 1])
        self.assertEqual(full.lengths, [2, 3, 4, 5, 6])
        self.assertEqual(sum(cached.lengths), 2 + 5 - 1)

    def test_full_window_is_rebuilt_when_absolute_positions_shift(self):
        model = self.make_model(context=3)
        counter = ProjectionCounter(model.network.layers[1].attention.key)
        with patch.object(nn.Linear, "__call__", autospec=True, side_effect=counter):
            actual = model.generate("ab", length=5, temperature=0.8, seed=11)
        self.assertEqual(counter.lengths, [2, 1, 3, 3, 3])
        self.assertEqual(actual, model.generate("ab", length=5, temperature=0.8, seed=11, use_cache=False))

    def test_each_generation_prefills_its_own_cache_and_zero_length_skips_network(self):
        model = self.make_model()
        counter = ProjectionCounter(model.network.layers[1].attention.key)
        with patch.object(nn.Linear, "__call__", autospec=True, side_effect=counter):
            first = model.generate("ab", length=3, temperature=0.8, seed=9)
            self.assertEqual(model.generate("ab", length=3, temperature=0.8, seed=9), first)
            self.assertEqual(model.generate("abcabc", length=0), "abcabc")
        self.assertEqual(counter.lengths, [2, 1, 1, 2, 1, 1])

    def test_generation_preserves_weights_gradients_global_random_and_context(self):
        model = self.make_model()
        nn.CrossEntropy()(model([[0, 1, 2]]), [[1, 2, 0]]).backward()
        parameters = model.parameters()
        gradients = [parameter.grad for parameter in parameters]
        gradient_values = [gradient.values[:] for gradient in gradients]
        weights = [parameter.values[:] for parameter in parameters]
        context, global_random = (is_recording(), is_training()), random.getstate()
        with set_recording(False), set_training(False):
            model.generate("ab", length=4, temperature=0.8, top_k=2, seed=12)
            self.assertFalse(is_recording())
            self.assertFalse(is_training())
        self.assertEqual((is_recording(), is_training()), context)
        self.assertEqual(random.getstate(), global_random)
        for parameter, gradient, saved_gradient, weight in zip(parameters, gradients, gradient_values, weights, strict=True):
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(parameter.grad.values, saved_gradient)
            self.assertEqual(parameter.values, weight)

    def test_weight_update_and_model_load_never_reuse_an_old_generation_cache(self):
        model = self.make_model()
        model.generate("ab", length=3)
        optimizer = nn.SGD(model, learning_rate=0.1)
        optimizer.zero_grad()
        nn.CrossEntropy()(model([[0, 1, 2]]), [[2, 0, 1]]).backward()
        optimizer.step()
        self.assertEqual(model.generate("ab", length=4, temperature=0.8, seed=13),
                         model.generate("ab", length=4, temperature=0.8, seed=13, use_cache=False))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            model.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"model", "parameters", "characters", "context_length"})
            restored = lm.Model().load(path)
            self.assertEqual(restored.generate("ab", length=4, temperature=0.8, seed=13),
                             model.generate("ab", length=4, temperature=0.8, seed=13))
            replacement = self.make_model(seed=9)
            replacement.save(path)
            model.load(path)
        self.assertEqual(model.generate("ab", length=4, temperature=0.8, seed=13),
                         replacement.generate("ab", length=4, temperature=0.8, seed=13, use_cache=False))


if __name__ == "__main__":
    unittest.main()
