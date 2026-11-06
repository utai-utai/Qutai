"""DiT 结构、时间编码、共享参数与原地加载。"""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, nn
from qutai.nn.dit import DiTBlock, TimeEmbedding
from tests.test_dit import activate


class DiTSerializationTests(unittest.TestCase):
    def make_model(self, seed=7):
        model = nn.DiT(1, (2, 4), patch_size=(1, 2), features=4, heads=2, depth=2, hidden_features=5, seed=seed)
        activate(model)
        model.blocks[0].attention.key.weight = model.blocks[0].attention.query.weight
        model.blocks[1].attention.query.weight = model.blocks[0].attention.query.weight
        return model

    def test_roundtrip_restores_layout_heads_conditions_shared_parameters_and_flags(self):
        model = self.make_model()
        model.position_embedding.weight.requires_grad = False
        inputs = Tensor([index / 10 for index in range(16)], shape=(2, 1, 2, 4))
        times = [0.2, 0.8]
        scores = model(inputs, times, train=False).tolist()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "dit.json"
            nn.Model([model]).save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"model", "parameters"})
            structure = data["model"]["layers"][0]
            self.assertEqual(structure["type"], "DiT")
            self.assertEqual(structure["image_shape"], [1, 2, 4])
            self.assertEqual(structure["patch_size"], [1, 2])
            restored = nn.Model().load(path).layers[0]
        self.assertIsInstance(restored, nn.DiT)
        self.assertIsInstance(restored.time_embedding, TimeEmbedding)
        self.assertTrue(all(isinstance(block, DiTBlock) for block in restored.blocks))
        self.assertEqual(restored.image_shape, (1, 2, 4))
        self.assertEqual(restored.patch_size, (1, 2))
        self.assertEqual([block.attention.heads for block in restored.blocks], [2, 2])
        self.assertTrue(all(not block.attention.causal for block in restored.blocks))
        self.assertEqual(restored.blocks[0].attention_norm.eps, 0.2)
        self.assertEqual(restored.time_embedding.features, 4)
        self.assertIs(restored.blocks[0].attention.query.weight, restored.blocks[0].attention.key.weight)
        self.assertIs(restored.blocks[1].attention.query.weight, restored.blocks[0].attention.query.weight)
        self.assertFalse(restored.position_embedding.weight.requires_grad)
        self.assertEqual(restored(inputs, times, train=False).tolist(), scores)
        for source, loaded in zip(model.parameters(), restored.parameters(), strict=True):
            self.assertEqual(source.values, loaded.values)
            self.assertEqual(source.requires_grad, loaded.requires_grad)

    def test_existing_load_preserves_model_parameter_optimizer_and_freeze_references(self):
        source, target = self.make_model(), self.make_model(seed=8)
        source.position_embedding.weight.requires_grad = False
        target.output_norm.weight.requires_grad = False
        container = nn.Model([target])
        optimizer = nn.SGD(container, learning_rate=0.1)
        parameters = target.parameters()
        flags = [parameter.requires_grad for parameter in parameters]
        inputs = Tensor([index / 10 for index in range(8)], shape=(1, 1, 2, 4))
        target(inputs, [0.3]).square().sum().backward()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "dit.json"
            nn.Model([source]).save(path)
            self.assertIs(container.load(path), container)
        self.assertIs(container.layers[0], target)
        for current, previous, optimized, flag in zip(target.parameters(), parameters, optimizer.parameters, flags, strict=True):
            self.assertIs(current, previous)
            self.assertIs(current, optimized)
            self.assertEqual(current.requires_grad, flag)
            self.assertIsNone(current.grad)
        self.assertEqual(target(inputs, [0.3], train=False).tolist(), source(inputs, [0.3], train=False).tolist())
        self.assertIs(target.blocks[0].attention.query.weight, target.blocks[1].attention.query.weight)

    def test_late_bad_parameter_and_changed_layout_reject_atomically(self):
        source, target = self.make_model(), self.make_model(seed=8)
        parameters = target.parameters()
        inputs = Tensor([index / 10 for index in range(8)], shape=(1, 1, 2, 4))
        target(inputs, [0.3]).sum().backward()
        values = [parameter.values[:] for parameter in parameters]
        gradients = tuple(parameter.grad for parameter in parameters)
        container = nn.Model([target])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "dit.json"
            nn.Model([source]).save(path)
            original = json.loads(path.read_text(encoding="utf-8"))
            for case in ("parameter", "layout", "heads"):
                data = copy.deepcopy(original)
                data["parameters"][0]["values"][0] += 123
                if case == "parameter": data["parameters"][-1]["shape"] = [999]
                elif case == "layout": data["model"]["layers"][0]["image_shape"] = [1, 4, 2]
                else: data["model"]["layers"][0]["blocks"][0]["attention"]["heads"] = 1
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(case=case), self.assertRaises(ValueError):
                    container.load(path)
                for parameter, current, value, gradient in zip(parameters, target.parameters(), values, gradients, strict=True):
                    self.assertIs(parameter, current)
                    self.assertEqual(current.values, value)
                    self.assertIs(current.grad, gradient)

    def test_parameterless_replacement_and_nondefault_norm_survive_roundtrip(self):
        source = nn.DiT(2, (2, 4), patch_size=(1, 2), features=4, heads=2, seed=7)
        activate(source)
        source.output_norm = nn.Identity()
        inputs = Tensor([index / 10 for index in range(16)], shape=(1, 2, 2, 4))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "dit.json"
            nn.Model([source]).save(path)
            restored = nn.Model().load(path).layers[0]
        self.assertIsInstance(restored.output_norm, nn.Identity)
        self.assertEqual(restored(inputs, [0.4], train=False).tolist(), source(inputs, [0.4], train=False).tolist())


if __name__ == "__main__":
    unittest.main()
