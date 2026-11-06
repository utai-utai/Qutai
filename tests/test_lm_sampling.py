"""续写采样的分布、随机状态、上下文与保存恢复。"""

import random
import unittest
from bisect import bisect
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qutai import Tensor, lm, nn
from qutai.core.context import is_recording, is_training, set_recording, set_training


class FixedScoresModel(lm.Model):
    """固定三类得分，只记录 generate 传入的上下文。"""
    def __init__(self, scores):
        self.tokenizer = lm.Tokenizer("abc")
        self.context_length = 2
        self.scores = scores
        self.calls = []

    def __call__(self, tokens, *, train=True):
        self.calls.append((tokens, train))
        return Tensor([[self.scores]])


class SamplingTests(unittest.TestCase):
    def make_model(self):
        return lm.Model(lm.Tokenizer("abc"), context_length=2, features=2, heads=1,
                        hidden_features=3, seed=7)

    def test_default_greedy_and_top_k_one_preserve_existing_selection(self):
        model = FixedScoresModel([1, 4, 2])
        self.assertEqual(model.generate("abcab", use_cache=False, length=4), "abcabbbbb")
        self.assertEqual(model.generate("abcab", use_cache=False, length=4, temperature=1, top_k=1), "abcabbbbb")
        self.assertEqual(model.calls[0], ([[0, 1]], False))
        self.assertTrue(all(len(tokens[0]) == 2 and not train for tokens, train in model.calls))

    def test_temperature_sampling_matches_independent_inverse_distribution(self):
        # 除以温度 2 后三类权重为 1:2:4，按得分从高到低采样。
        model = FixedScoresModel([0, log(4), log(16)])
        reference = random.Random(23)
        expected = "a" + "".join("cba"[bisect([4 / 7, 6 / 7], reference.random())] for _ in range(80))
        self.assertEqual(model.generate("a", use_cache=False, length=80, temperature=2, seed=23), expected)

    def test_top_k_filters_before_sampling_and_large_k_includes_entire_vocabulary(self):
        model = FixedScoresModel([0, log(2), log(4)])
        reference = random.Random(29)
        expected = "a" + "".join("cb"[reference.random() >= 2 / 3] for _ in range(80))
        self.assertEqual(model.generate("a", use_cache=False, length=80, temperature=1, top_k=2, seed=29), expected)
        self.assertEqual(model.generate("a", use_cache=False, length=40, temperature=1, top_k=99, seed=8),
                         model.generate("a", use_cache=False, length=40, temperature=1, seed=8))

    def test_seed_reproducibility_random_diversity_and_global_random_untouched(self):
        model = FixedScoresModel([0, 0, 0])
        global_state = random.getstate()
        first = model.generate("a", use_cache=False, length=30, temperature=1, seed=42)
        self.assertEqual(model.generate("a", use_cache=False, length=30, temperature=1, seed=42), first)
        self.assertNotEqual(model.generate("a", use_cache=False, length=30, temperature=1, seed=43), first)
        model.generate("a", use_cache=False, length=10, temperature=1)
        self.assertEqual(random.getstate(), global_state)

    def test_default_seed_is_none_and_each_call_owns_its_random_generator(self):
        model = FixedScoresModel([1, 2, 3])
        with patch("qutai.lm.model.Random", wraps=random.Random) as constructor:
            model.generate("a", use_cache=False, length=2, temperature=1)
            model.generate("a", use_cache=False, length=2, temperature=1)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(constructor.call_args_list[0].args, (None,))
        self.assertEqual(constructor.call_args_list[1].args, (None,))

    def test_stable_probabilities_for_large_scores_and_tiny_positive_temperature(self):
        model = FixedScoresModel([1e308, 1e308, -1e308])
        generated = model.generate("a", use_cache=False, length=30, temperature=0.5, seed=1)
        self.assertLessEqual(set(generated), {"a", "b"})
        model.scores = [1, 2, 3]
        self.assertEqual(model.generate("a", use_cache=False, length=5, temperature=5e-324, seed=2), "accccc")

    def test_zero_length_preserves_prompt_without_running_network(self):
        model = FixedScoresModel([1, 2, 3])
        self.assertEqual(model.generate("abcabc", use_cache=False, length=0, temperature=1, top_k=2), "abcabc")
        self.assertEqual(model.calls, [])

    def test_invalid_sampling_parameters_fail_clearly(self):
        model = FixedScoresModel([1, 2, 3])
        for temperature in (-1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(ValueError, "采样温度"):
                model.generate("a", use_cache=False, length=1, temperature=temperature)
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "top_k"):
                model.generate("a", use_cache=False, length=1, temperature=1, top_k=top_k)
        self.assertEqual(model.calls, [])

    def test_sampled_generation_preserves_training_gradients_and_context(self):
        model = self.make_model()
        nn.CrossEntropy()(model([[0, 1]]), [[1, 2]]).backward()
        parameters = model.parameters()
        gradients = tuple(parameter.grad for parameter in parameters)
        saved_gradients = [gradient.values[:] for gradient in gradients]
        saved_values = [parameter.values[:] for parameter in parameters]
        before = is_recording(), is_training()
        with set_recording(False), set_training(False):
            model.generate("abc", length=4, temperature=0.8, top_k=2, seed=1)
            self.assertFalse(is_recording())
            self.assertFalse(is_training())
        self.assertEqual((is_recording(), is_training()), before)
        for parameter, gradient, values, saved in zip(parameters, gradients, saved_values, saved_gradients, strict=True):
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(parameter.grad.values, saved)
            self.assertEqual(parameter.values, values)

    def test_save_load_preserves_seeded_sampling(self):
        source = self.make_model()
        expected = source.generate("abc", length=8, temperature=0.9, top_k=2, seed=17)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            restored = lm.Model().load(path)
        self.assertEqual(restored.generate("abc", length=8, temperature=0.9, top_k=2, seed=17), expected)


if __name__ == "__main__":
    unittest.main()
