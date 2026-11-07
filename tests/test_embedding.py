import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, nn
from qutai.core.tensor import embedding
from qutai.core.shape import flatten_indices


class EmbeddingTests(unittest.TestCase):
    def test_lookup_preserves_scalar_and_nested_index_shapes(self):
        weight = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        scalar = embedding(2, weight)
        self.assertEqual(scalar.shape, (2,))
        self.assertEqual(scalar.tolist(), [5, 6])
        result = embedding(((2, 0), (1, 2)), weight)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.tolist(), [[[5, 6], [1, 2]], [[3, 4], [5, 6]]])
        self.assertEqual(result.parents, (weight,))
        self.assertIs(result.parents[0], weight)
        self.assertEqual(embedding([1, 0], [[1, 2], [3, 4]]), [[3, 4], [1, 2]])

    def test_repeated_indices_accumulate_each_feature_gradient(self):
        weight = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        result = embedding([[2, 0, 2], [2, 1, 0]], weight)
        result.backward([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        self.assertEqual(weight.grad.tolist(), [[14, 16], [9, 10], [13, 16]])

    def test_weight_gradients_match_central_differences(self):
        weight = Tensor([[0.2, -0.4, 0.7], [0.1, 0.8, -0.2], [0.6, -0.3, 0.9], [-0.5, 0.4, 0.3]], requires_grad=True)
        indices = ((2, 0), (2, 1))
        embedding(indices, weight).square().mean().backward()
        epsilon = 1e-6
        for index, analytic in enumerate(weight.grad.values):
            original = weight.values[index]
            weight.values[index] = original + epsilon
            upper = embedding(indices, weight).square().mean().item()
            weight.values[index] = original - epsilon
            lower = embedding(indices, weight).square().mean().item()
            weight.values[index] = original
            self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=7)
        self.assertEqual(weight.grad.tolist()[3], [0, 0, 0])

    def test_index_flattening_preserves_large_integers_exactly(self):
        large = 2 ** 80 + 1
        result = flatten_indices(((large, 0), (2, 3)))
        self.assertEqual(result, [large, 0, 2, 3])
        self.assertTrue(all(type(index) is int for index in result))
        self.assertEqual(flatten_indices(large), [large])
        self.assertEqual(flatten_indices([]), [])

    def test_floats_tensors_and_booleans_are_not_integer_indices(self):
        layer = nn.Embedding(3, 2, seed=7)
        for indices in (1.0, [0, 1.0], Tensor([0, 1]), True, [False], "1"):
            with self.subTest(indices=indices), self.assertRaises(TypeError):
                layer(indices)
            with self.subTest(flatten=indices), self.assertRaises(TypeError):
                flatten_indices(indices)

    def test_negative_and_out_of_range_indices_raise(self):
        layer = nn.Embedding(3, 2, seed=7)
        for indices in (-1, 3, [0, 3], [[0, 1], [2, -1]], 2 ** 80 + 1):
            with self.subTest(indices=indices), self.assertRaisesRegex(IndexError, "范围"):
                layer(indices)

    def test_ragged_indices_raise_before_lookup(self):
        with self.assertRaisesRegex(ValueError, "形状"):
            embedding([[0], [1, 2]], Tensor([[1, 2], [3, 4], [5, 6]]))

    def test_empty_index_batches_keep_shape_and_have_zero_weight_gradient(self):
        for indices, shape in (([], (0, 2)), ([[], []], (2, 0, 2))):
            with self.subTest(indices=indices):
                weight = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
                result = embedding(indices, weight)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.values, [])
                result.sum().backward()
                self.assertEqual(weight.grad.tolist(), [[0, 0], [0, 0], [0, 0]])

    def test_initialization_is_reproducible_without_touching_global_random(self):
        state = random.getstate()
        first, second = nn.Embedding(3, 2, seed=7), nn.Embedding(3, 2, seed=7)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first.weight.shape, (3, 2))
        self.assertEqual(first.weight.values, second.weight.values)
        self.assertNotEqual(first.weight.values, nn.Embedding(3, 2, seed=8).weight.values)
        self.assertTrue(first.weight.requires_grad)
        self.assertEqual(first.weight.parents, ())
        self.assertIs(first.parameters()[0], first.weight)

    def test_invalid_counts_features_and_weight_shapes(self):
        for count, features in ((0, 2), (-1, 2), (3, 0), (3, -1)):
            with self.subTest(count=count, features=features), self.assertRaises(ValueError):
                nn.Embedding(count, features)
        for weight in ([1, 2], [[[1, 2]]], [], Tensor([], shape=(2, 0))):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                embedding([], weight)

    def test_empty_model_load_restores_embedding_and_shared_frozen_weight(self):
        first, second = nn.Embedding(3, 2, seed=7), nn.Embedding(3, 2, seed=8)
        second.weight = first.weight
        first.weight.requires_grad = False
        source = nn.Model([first, second])
        indices = [[2, 0], [1, 2]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "embedding.json"
            source.save(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(record), {"model", "parameters"})
            self.assertEqual(len(record["parameters"]), 1)
            restored = nn.Model().load(path)
        self.assertIsInstance(restored.layers[0], nn.Embedding)
        self.assertIs(restored.layers[0].weight, restored.layers[1].weight)
        self.assertFalse(restored.layers[0].weight.requires_grad)
        self.assertEqual(restored.layers[0](indices).tolist(), first(indices).tolist())

    def test_in_place_load_preserves_weight_and_optimizer_references(self):
        source = nn.Model([nn.Embedding(3, 2, seed=7)])
        target = nn.Model([nn.Embedding(3, 2, seed=8)])
        weight = target.layers[0].weight
        optimizer = nn.SGD(target)
        target.layers[0]([0, 2, 2]).sum().backward()
        self.assertIsNotNone(weight.grad)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "embedding.json"
            source.save(path)
            target.load(path)
        self.assertIs(target.layers[0].weight, weight)
        self.assertIs(optimizer.parameters[0], weight)
        self.assertIsNone(weight.grad)
        self.assertEqual(target.layers[0]([1, 0]).tolist(), source.layers[0]([1, 0]).tolist())


if __name__ == "__main__":
    unittest.main()
