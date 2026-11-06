"""参数保存、加载、结构匹配与训练续接。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import nn


def make_model(seed: int) -> nn.Model:
    return nn.Model([nn.Linear(1, 3, seed=seed), nn.ReLU(), nn.Linear(3, 1, seed=seed + 1)])


class SerializationTests(unittest.TestCase):
    def test_empty_model_load_is_atomic_and_can_resume_training(self):
        source = make_model(42)
        restored = nn.Model()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = path.read_text(encoding="utf-8")
            invalid = json.loads(saved)
            invalid["parameters"][-1]["values"] = ["invalid"]
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                restored.load(path)
            self.assertEqual(restored.layers, ())
            self.assertEqual(restored.parameters(), ())
            path.write_text(saved, encoding="utf-8")
            self.assertIs(restored.load(path), restored)
        optimizer = nn.SGD(restored, learning_rate=0.01)
        inputs, targets = [[-1], [0], [1]], [[1], [1], [1]]
        loss = nn.MSE()(restored(inputs), targets)
        loss.backward()
        optimizer.step()
        self.assertLess(nn.MSE()(restored(inputs, train=False), targets).item(), loss.item())

    def test_roundtrip_preserves_predictions_and_values(self):
        source = make_model(42)
        inputs = [[-1], [0], [0.5], [1]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"model", "parameters"})
            self.assertEqual([layer["type"] for layer in data["model"]["layers"]], ["Linear", "ReLU", "Linear"])
            self.assertEqual(data["parameters"][0]["shape"], [1, 3])
            self.assertEqual(data["parameters"][0]["values"], source.parameters()[0].values)
            restored = nn.Model().load(str(path))
        self.assertIsInstance(restored, nn.Model)
        self.assertIsInstance(restored.layers[0], nn.Linear)
        self.assertIsInstance(restored.layers[1], nn.ReLU)
        self.assertIsInstance(restored.layers[2], nn.Linear)
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())
        for original, loaded in zip(source.parameters(), restored.parameters(), strict=True):
            self.assertEqual(original.shape, loaded.shape)
            self.assertEqual(original.values, loaded.values)
            self.assertIsNot(original.values, loaded.values)

    def test_load_preserves_optimizer_references_and_resumes_training(self):
        source = make_model(42)
        restored = make_model(7)
        parameters = restored.parameters()
        parameters[0].requires_grad = False
        flags = [parameter.requires_grad for parameter in parameters]
        optimizer = nn.SGD(restored, learning_rate=0.01)
        inputs, targets = [[-1], [0], [1]], [[1], [1], [1]]
        nn.MSE()(restored(inputs), targets).backward()
        self.assertIsNotNone(parameters[-1].grad)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            self.assertIs(restored.load(path), restored)
        for previous, current in zip(optimizer.parameters, restored.parameters(), strict=True):
            self.assertIs(previous, current)
            self.assertIsNone(current.grad)
            self.assertEqual(current.parents, ())
        self.assertEqual([parameter.requires_grad for parameter in parameters], flags)
        loss = nn.MSE()(restored(inputs), targets)
        loss.backward()
        optimizer.step()
        self.assertLess(nn.MSE()(restored(inputs, train=False), targets).item(), loss.item())

    def test_parameter_count_mismatch_preserves_model(self):
        source = nn.Model([nn.Linear(1, 3, seed=42)])
        restored = make_model(7)
        before = [parameter.values[:] for parameter in restored.parameters()]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            with self.assertRaisesRegex(ValueError, "模型结构"):
                restored.load(path)
        self.assertEqual([parameter.values for parameter in restored.parameters()], before)

    def test_invalid_later_parameter_does_not_partly_load(self):
        source = make_model(42)
        restored = make_model(7)
        nn.MSE()(restored([[1]]), [[1]]).backward()
        parameters = restored.parameters()
        before = [parameter.values[:] for parameter in parameters]
        gradients = [parameter.grad for parameter in parameters]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = path.read_text(encoding="utf-8")
            for key, invalid in (("shape", [1, 1]), ("values", [1, 2]), ("values", ["invalid"])):
                with self.subTest(key=key, invalid=invalid):
                    data = json.loads(saved)
                    data["parameters"][-1][key] = invalid
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises((TypeError, ValueError)):
                        restored.load(path)
                    self.assertEqual([parameter.values for parameter in parameters], before)
                    for parameter, gradient in zip(parameters, gradients, strict=True):
                        self.assertIs(parameter.grad, gradient)

    def test_nested_shared_parameters_are_saved_once(self):
        shared = nn.Linear(1, 1, seed=42)
        source = nn.Model([shared, nn.Model([nn.ReLU(), shared])])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))["parameters"]), 2)
            restored = nn.Model()
            restored.load(path)
        self.assertIsInstance(restored.layers[1], nn.Model)
        self.assertEqual(len(restored.parameters()), 2)
        for first, nested in zip(restored.layers[0].parameters(), restored.layers[1].parameters(), strict=True):
            self.assertIs(first, nested)
        inputs = [[-1], [0], [1]]
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_same_parameter_shapes_do_not_hide_different_layers(self):
        source = make_model(42)
        wrong = nn.Model([nn.Linear(1, 3, seed=7), nn.Linear(3, 1, seed=8)])
        before = [parameter.values[:] for parameter in wrong.parameters()]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            with self.assertRaisesRegex(ValueError, "模型结构"):
                wrong.load(path)
        self.assertEqual([parameter.values for parameter in wrong.parameters()], before)

    def test_distinct_layers_keep_tied_weights(self):
        first = nn.Linear(2, 2, seed=42)
        last = nn.Linear(2, 2, seed=43)
        last.weight = first.weight
        source = nn.Model([first, nn.ReLU(), last])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model()
            restored.load(path)
        self.assertEqual(len(restored.parameters()), 3)
        self.assertIs(restored.layers[0].parameters()[0], restored.layers[2].parameters()[0])
        self.assertIsNot(restored.layers[0].parameters()[1], restored.layers[2].parameters()[1])
        inputs = [[1, -2], [3, 4]]
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_automatic_load_preserves_trainable_flags(self):
        source = make_model(42)
        source.parameters()[0].requires_grad = False
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model()
            restored.load(path)
        self.assertEqual([parameter.requires_grad for parameter in restored.parameters()], [False, True, True, True])
        nn.MSE()(restored([[1]]), [[1]]).backward()
        self.assertIsNone(restored.parameters()[0].grad)
        self.assertIsNotNone(restored.parameters()[-1].grad)

    def test_legacy_parameter_file_can_be_loaded_into_existing_model(self):
        source = make_model(42)
        restored = make_model(7)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            records = [{key: record[key] for key in ("shape", "values")} for record in data["parameters"]]
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少模型结构"):
                nn.Model().load(path)
            restored.load(path)
            restored.save(path)
            automatic = nn.Model()
            automatic.load(path)
        inputs = [[-1], [0], [1]]
        self.assertEqual(automatic(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_invalid_model_structure_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            make_model(42).save(path)
            saved = path.read_text(encoding="utf-8")
            for issue in ("type", "index", "negative_index", "config", "unused"):
                with self.subTest(issue=issue):
                    data = json.loads(saved)
                    if issue == "type": data["model"]["layers"][1]["type"] = "Unknown"
                    elif issue == "index": data["model"]["layers"][0]["parameters"][0] = len(data["parameters"])
                    elif issue == "negative_index": data["model"]["layers"][0]["parameters"][0] = -1
                    elif issue == "config": data["model"]["layers"][0]["in_features"] = 2
                    else: data["parameters"].append(data["parameters"][0])
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        nn.Model().load(path)

    def test_models_without_parameters_keep_their_layers(self):
        inputs = [[-1, 0, 2]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            for source in (nn.Model([]), nn.Model([nn.ReLU()])):
                with self.subTest(layers=len(source.layers)):
                    source.save(path)
                    restored = nn.Model()
                    restored.load(path)
                    self.assertEqual(len(restored.layers), len(source.layers))
                    self.assertEqual(restored.parameters(), ())
                    self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())


if __name__ == "__main__":
    unittest.main()
