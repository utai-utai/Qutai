"""字符语言模型的因果计算、参数梯度、续写和完整持久化。"""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, lm, nn
from qutai.core.context import is_recording, is_training, set_recording, set_training


class LanguageModelTests(unittest.TestCase):
    def make_model(self, characters="ab", context=2, seed=7):
        return lm.Model(lm.Tokenizer(characters), context_length=context, features=2, heads=1,
                        hidden_features=3, depth=1, seed=seed)

    def test_integer_input_and_all_small_model_gradients(self):
        model = self.make_model()
        inputs, targets = [[0, 1]], [[1, 0]]
        scores = model(inputs)
        self.assertEqual(scores.shape, (1, 2, 2))
        self.assertTrue(scores.requires_grad)
        self.assertIs(model.layers[0], model.embedding)
        self.assertEqual(model.layers[1:], model.network.layers)
        loss_function = nn.CrossEntropy()
        loss_function(scores, targets).backward()
        epsilon = 1e-6
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            for index, analytic in enumerate(parameter.grad.values):
                with self.subTest(shape=parameter.shape, index=index):
                    original = parameter.values[index]
                    parameter.values[index] = original + epsilon
                    upper = loss_function(model(inputs, train=False), targets).item()
                    parameter.values[index] = original - epsilon
                    lower = loss_function(model(inputs, train=False), targets).item()
                    parameter.values[index] = original
                    self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=5)

    def test_future_tokens_cannot_change_prefix_or_receive_its_gradient(self):
        model = self.make_model("abcd", context=3)
        first = model([[0, 1, 2]])
        changed = model([[0, 1, 3]])
        self.assertEqual(first[:, :2].tolist(), changed[:, :2].tolist())
        first[:, :2].square().sum().backward()
        self.assertEqual(model.embedding.weight.grad.tolist()[2:], [[0, 0], [0, 0]])
        self.assertTrue(any(value != 0 for row in model.embedding.weight.grad.tolist()[:2] for value in row))

    def test_generate_crops_context_but_preserves_the_whole_prompt(self):
        model = self.make_model("abc", context=2)
        prompt = "abcab"
        expected = model.tokenizer.encode(prompt)
        for _ in range(3):
            expected.append(model([expected[-2:]], train=False)[0, -1].argmax())
        generated = model.generate(prompt, length=3)
        self.assertEqual(generated, model.tokenizer.decode(expected))
        self.assertTrue(generated.startswith(prompt))
        self.assertEqual(len(generated), len(prompt) + 3)
        self.assertEqual(model.generate(prompt, length=0), prompt)

    def test_generate_preserves_old_gradients_and_restores_calling_context(self):
        model = self.make_model()
        nn.CrossEntropy()(model([[0, 1]]), [[1, 0]]).backward()
        parameters = model.parameters()
        gradients = tuple(parameter.grad for parameter in parameters)
        values = [gradient.values[:] for gradient in gradients]
        weights = [parameter.values[:] for parameter in parameters]
        recording, training = is_recording(), is_training()
        first = model.generate("ab", length=2)
        self.assertEqual((is_recording(), is_training()), (recording, training))
        with set_recording(False), set_training(False):
            self.assertEqual(model.generate("ab", length=2), first)
            self.assertFalse(is_recording())
            self.assertFalse(is_training())
        self.assertEqual((is_recording(), is_training()), (recording, training))
        for parameter, gradient, expected_gradient, expected_weight in zip(parameters, gradients, values, weights, strict=True):
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(parameter.grad.values, expected_gradient)
            self.assertEqual(parameter.values, expected_weight)

    def test_generate_rejects_unknown_characters_empty_prompt_and_negative_length(self):
        model = self.make_model()
        for prompt, length in (("", 1), ("ab", -1), ("c", 1), ("c", 0)):
            with self.subTest(prompt=prompt, length=length), self.assertRaises(ValueError):
                model.generate(prompt, length=length)

    def test_eval_has_identical_values_and_no_graph(self):
        model = self.make_model()
        training, evaluation = model([[0, 1]]), model([[0, 1]], train=False)
        self.assertEqual(training.tolist(), evaluation.tolist())
        self.assertFalse(evaluation.requires_grad)
        self.assertEqual(evaluation.parents, ())
        self.assertTrue(model([[0, 1]]).requires_grad)

    def test_empty_load_restores_vocabulary_context_frozen_and_shared_parameters(self):
        source = self.make_model("乙 a\n甲😊乙", context=3)
        attention = source.network.layers[1].attention
        attention.key.weight = attention.query.weight
        source.network.layers[0].weight.requires_grad = False
        inputs = [source.tokenizer.encode("乙 a")]
        scores = source(inputs, train=False).tolist()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"model", "parameters", "characters", "context_length"})
            self.assertEqual(data["characters"], "乙 a\n甲😊")
            restored = lm.Model().load(path)
        self.assertEqual(restored.context_length, 3)
        self.assertEqual(restored.tokenizer.characters, source.tokenizer.characters)
        self.assertEqual(restored.tokenizer.indices, source.tokenizer.indices)
        restored_attention = restored.network.layers[1].attention
        self.assertTrue(restored_attention.causal)
        self.assertIs(restored_attention.key.weight, restored_attention.query.weight)
        self.assertFalse(restored.network.layers[0].weight.requires_grad)
        for original, loaded in zip(source.parameters(), restored.parameters(), strict=True):
            self.assertEqual(loaded.requires_grad, original.requires_grad)
        self.assertEqual(restored(inputs, train=False).tolist(), scores)
        self.assertEqual(restored.generate("乙", length=3), source.generate("乙", length=3))

    def test_existing_load_preserves_parameter_optimizer_and_freeze_choices(self):
        source, target = self.make_model(), self.make_model(seed=8)
        source.embedding.weight.requires_grad = False
        source_attention, target_attention = source.network.layers[1].attention, target.network.layers[1].attention
        source_attention.key.weight = source_attention.query.weight
        target_attention.key.weight = target_attention.query.weight
        target.network.layers[0].weight.requires_grad = False
        parameters = target.parameters()
        flags = [parameter.requires_grad for parameter in parameters]
        embedding, network, tokenizer = target.embedding, target.network, target.tokenizer
        optimizer = nn.SGD(target, learning_rate=0.1)
        nn.CrossEntropy()(target([[0, 1]]), [[1, 0]]).backward()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            self.assertIs(target.load(path), target)
        self.assertIs(target.embedding, embedding)
        self.assertIs(target.network, network)
        self.assertIs(target.tokenizer, tokenizer)
        for current, original, optimized, flag in zip(target.parameters(), parameters, optimizer.parameters, flags, strict=True):
            self.assertIs(current, original)
            self.assertIs(current, optimized)
            self.assertEqual(current.requires_grad, flag)
            self.assertIsNone(current.grad)
        self.assertEqual(target([[0, 1]], train=False).tolist(), source([[0, 1]], train=False).tolist())
        self.assertIs(target_attention.query.weight, target_attention.key.weight)

    def test_inconsistent_metadata_rejects_without_mutating_live_model(self):
        source, target = self.make_model(), self.make_model(seed=8)
        nn.CrossEntropy()(target([[0, 1]]), [[1, 0]]).backward()
        parameters, tokenizer, embedding, network = target.parameters(), target.tokenizer, target.embedding, target.network
        weights = [parameter.values[:] for parameter in parameters]
        gradients = tuple(parameter.grad for parameter in parameters)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            original = json.loads(path.read_text(encoding="utf-8"))
            for case in ("order", "length", "size", "duplicate", "missing"):
                data = copy.deepcopy(original)
                for parameter in data["parameters"]:
                    parameter["values"] = [value + 123 for value in parameter["values"]]
                if case == "order": data["characters"] = "ba"
                elif case == "length": data["context_length"] = 3
                elif case == "size": data["characters"] = "abc"
                elif case == "duplicate": data["characters"] = "aab"
                else: del data["characters"]
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(case=case), self.assertRaises(ValueError):
                    target.load(path)
                self.assertIs(target.tokenizer, tokenizer)
                self.assertIs(target.embedding, embedding)
                self.assertIs(target.network, network)
                self.assertEqual(target.context_length, 2)
                for current, parameter, weight, gradient in zip(target.parameters(), parameters, weights, gradients, strict=True):
                    self.assertIs(current, parameter)
                    self.assertEqual(current.values, weight)
                    self.assertIs(current.grad, gradient)

    def test_bad_network_parameter_rejects_before_partial_in_place_load(self):
        source, target = self.make_model(), self.make_model(seed=8)
        parameters = target.parameters()
        weights = [parameter.values[:] for parameter in parameters]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["parameters"][0]["values"][0] += 123
            data["parameters"][-1]["shape"] = [999]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                target.load(path)
        for parameter, weight in zip(parameters, weights, strict=True):
            self.assertEqual(parameter.values, weight)

    def test_empty_model_stays_empty_after_bad_metadata_load(self):
        source, empty = self.make_model(), lm.Model()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "language.json"
            source.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["context_length"] = 3
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                empty.load(path)
        self.assertIsNone(empty.embedding)
        self.assertIsNone(empty.tokenizer)
        self.assertEqual(empty.network.layers, ())
        self.assertEqual(empty.layers, ())
        self.assertEqual(empty.parameters(), ())

    def test_invalid_inputs_and_uninitialized_model_fail_clearly(self):
        model = self.make_model()
        for inputs in ([0, 1], [], [[]], [[[0, 1]]], [[0, 1, 0]]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                model(inputs)
        for inputs in (Tensor([[0, 1]]), [[0.0, 1]]):
            with self.subTest(inputs=inputs), self.assertRaises(TypeError):
                model(inputs)
        with self.assertRaises(ValueError):
            lm.Model()([[0]])
        with self.assertRaises(ValueError):
            lm.Model().generate("a")
        with self.assertRaises(ValueError):
            lm.Model(lm.Tokenizer(""))


if __name__ == "__main__":
    unittest.main()
