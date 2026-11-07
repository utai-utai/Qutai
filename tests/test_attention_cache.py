import unittest
from unittest.mock import patch

from qutai import Tensor, nn
from qutai.core.context import is_recording, set_recording
from qutai.nn.cache import KVCache


class AttentionCacheTests(unittest.TestCase):
    def inputs(self):
        return Tensor([[[0.1 * (batch * 20 + row * 4 + feature) for feature in range(4)]
                        for row in range(5)] for batch in range(2)], requires_grad=True)

    def assert_values_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        for left, right in zip(actual.values, expected.values, strict=True):
            self.assertAlmostEqual(left, right, places=12)

    def test_batched_multihead_prefill_and_chunks_match_full_causal_attention(self):
        layer, inputs, cache = nn.MultiHeadAttention(4, heads=2, causal=True, seed=7), self.inputs(), KVCache()
        with set_recording(False):
            expected = layer(inputs)
        for start, stop in ((0, 2), (2, 4), (4, 5)):
            with self.subTest(start=start):
                actual = layer(inputs[:, start:stop], cache=cache)
                self.assert_values_close(actual, expected[:, start:stop])
                self.assertEqual(cache.length, stop)
                self.assertEqual(cache.keys.shape, (2, 2, stop, 2))
                self.assertEqual(cache.values.shape, cache.keys.shape)
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.parents, ())

    def test_each_projection_processes_only_new_tokens(self):
        layer, inputs, cache = nn.MultiHeadAttention(4, heads=2, causal=True, seed=7), self.inputs(), KVCache()
        with patch.object(nn.Linear, "__call__", autospec=True, side_effect=nn.Linear.__call__) as project:
            layer(inputs[:, :3], cache=cache)
            layer(inputs[:, 3:4], cache=cache)
            layer(inputs[:, 4:], cache=cache)
        for projection in (layer.query, layer.key, layer.value, layer.output):
            lengths = [call.args[1].shape[1] for call in project.call_args_list if call.args[0] is projection]
            self.assertEqual(lengths, [3, 1, 1])

    def test_cache_copies_values_and_appends_each_batch_and_head_separately(self):
        cache = KVCache()
        keys = Tensor([1, 2, 3, 4], shape=(2, 2, 1, 1), requires_grad=True)
        values = Tensor([10, 20, 30, 40], shape=keys.shape, requires_grad=True)
        cache.update(keys, values)
        keys.values[0], values.values[0] = 999, 999
        combined_keys, combined_values = cache.update(Tensor([5, 6, 7, 8], shape=keys.shape), Tensor([50, 60, 70, 80], shape=keys.shape))
        self.assertEqual(combined_keys.values, [1, 5, 2, 6, 3, 7, 4, 8])
        self.assertEqual(combined_values.values, [10, 50, 20, 60, 30, 70, 40, 80])
        self.assertEqual(cache.length, 2)
        self.assertFalse(combined_keys.requires_grad)
        self.assertEqual(combined_keys.parents, ())

    def test_transformer_block_cached_outputs_match_full_and_do_not_change_old_gradients(self):
        block, inputs, cache = nn.TransformerBlock(4, heads=2, causal=True, seed=9), self.inputs(), KVCache()
        block(inputs).square().sum().backward()
        parameters = block.parameters()
        gradients = tuple(parameter.grad for parameter in parameters)
        weights = [parameter.values[:] for parameter in parameters]
        saved_gradients = [gradient.values[:] for gradient in gradients]
        with set_recording(False):
            expected = block(inputs)
        recording = is_recording()
        self.assert_values_close(block(inputs[:, :2], cache=cache), expected[:, :2])
        actual = block(inputs[:, 2:], cache=cache)
        self.assert_values_close(actual, expected[:, 2:])
        self.assertEqual(is_recording(), recording)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(actual.parents, ())
        for parameter, gradient, weight, values in zip(parameters, gradients, weights, saved_gradients, strict=True):
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(parameter.grad.values, values)
            self.assertEqual(parameter.values, weight)

    def test_normal_attention_and_block_still_build_training_graph_after_cached_call(self):
        for layer in (nn.MultiHeadAttention(4, heads=2, causal=True, seed=7), nn.TransformerBlock(4, heads=2, causal=True, seed=7)):
            inputs = self.inputs()
            layer(inputs[:, :1], cache=KVCache())
            result = layer(inputs)
            self.assertTrue(result.requires_grad)
            result.square().sum().backward()
            self.assertIsNotNone(inputs.grad)
            self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))

    def test_position_offset_selects_correct_rows_and_preserves_training_gradient(self):
        layer = nn.PositionEmbedding(5, 2, seed=7)
        layer.weight.values = list(range(10))
        inputs = Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        result = layer(inputs, offset=2)
        self.assertEqual(result.tolist(), [[[4, 5], [6, 7]], [[4, 5], [6, 7]]])
        result.sum().backward()
        self.assertEqual(layer.weight.grad.tolist(), [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
        self.assertEqual(layer(inputs).tolist(), [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    def test_noncausal_cache_and_incompatible_cache_shapes_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "因果"):
            nn.MultiHeadAttention(4, heads=2)(self.inputs(), cache=KVCache())
        cache = KVCache()
        cache.update(Tensor([1, 2], shape=(1, 1, 1, 2)), Tensor([3, 4], shape=(1, 1, 1, 2)))
        with self.assertRaisesRegex(ValueError, "缓存一致"):
            cache.update(Tensor([1, 2], shape=(1, 2, 1, 1)), Tensor([3, 4], shape=(1, 2, 1, 1)))
        self.assertEqual(cache.length, 1)
        self.assertEqual(cache.keys.values, [1, 2])

    def test_position_offset_rejects_negative_and_overflowing_positions(self):
        layer = nn.PositionEmbedding(3, 2, seed=7)
        inputs = Tensor([[[1, 2], [3, 4]]])
        for offset in (-1, 2):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "位置"):
                layer(inputs, offset=offset)


if __name__ == "__main__":
    unittest.main()
