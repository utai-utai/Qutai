"""残差模型、BatchNorm 运行统计和共享参数的保存与训练续接。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import nn


def make_model(seed):
    branch = nn.Conv2d(2, 2, 1, bias=False, seed=seed)
    shortcut = nn.Conv2d(2, 2, 1, bias=False, seed=seed + 1)
    shortcut.weight = branch.weight
    block = nn.Residual(nn.Model([branch, nn.BatchNorm2d(2, eps=0.01, momentum=0.25)]),
                        shortcut=nn.Model([shortcut, nn.BatchNorm2d(2, eps=0.02, momentum=0.4)]))
    return nn.Model([block, nn.ReLU(), nn.GlobalAvgPool2d(), nn.Linear(2, 1, seed=seed + 2)])


class ResNetSerializationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [[[[1, 3]], [[2, 4]]], [[[5, 7]], [[6, 8]]]]
        self.targets = [[0.3], [-0.2]]

    def test_empty_load_restores_configuration_statistics_and_shared_weights(self):
        source = make_model(42)
        source(self.inputs)
        prediction = source(self.inputs, train=False).tolist()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved), {"model", "parameters"})
            self.assertEqual(len(saved["parameters"]), 7)
            restored = nn.Model().load(path)
        block = restored.layers[0]
        self.assertIsInstance(block, nn.Residual)
        self.assertIsInstance(restored.layers[2], nn.GlobalAvgPool2d)
        self.assertIs(block.branch.layers[0].weight, block.shortcut.layers[0].weight)
        for original, loaded in ((source.layers[0].branch, block.branch),
                                 (source.layers[0].shortcut, block.shortcut)):
            first, second = original.layers[1], loaded.layers[1]
            self.assertEqual((second.eps, second.momentum), (first.eps, first.momentum))
            self.assertEqual(second.running_mean, first.running_mean)
            self.assertEqual(second.running_var, first.running_var)
            self.assertIsNot(second.running_mean, first.running_mean)
        self.assertEqual(restored(self.inputs, train=False).tolist(), prediction)
        self.assertFalse(restored(self.inputs, train=False).requires_grad)

        before = [parameter.values[:] for parameter in restored.parameters()]
        optimizer = nn.SGD(restored, learning_rate=0.01)
        nn.MSE()(restored(self.inputs), self.targets).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in restored.parameters()))
        optimizer.step()
        self.assertTrue(any(parameter.values != values for parameter, values in zip(restored.parameters(), before, strict=True)))

    def test_existing_load_preserves_parameter_and_running_list_references(self):
        source, restored = make_model(42), make_model(7)
        source(self.inputs)
        optimizer = nn.SGD(restored, learning_rate=0.01)
        nn.MSE()(restored(self.inputs), self.targets).backward()
        block = restored.layers[0]
        norms = (block.branch.layers[1], block.shortcut.layers[1])
        buffers = [(layer.running_mean, layer.running_var) for layer in norms]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            self.assertIs(restored.load(path), restored)
        for original, current in zip(optimizer.parameters, restored.parameters(), strict=True):
            self.assertIs(current, original)
            self.assertIsNone(current.grad)
        current_norms = (restored.layers[0].branch.layers[1], restored.layers[0].shortcut.layers[1])
        source_norms = (source.layers[0].branch.layers[1], source.layers[0].shortcut.layers[1])
        for layer, (means, variances), original in zip(current_norms, buffers, source_norms, strict=True):
            self.assertIs(layer.running_mean, means)
            self.assertIs(layer.running_var, variances)
            self.assertEqual(means, original.running_mean)
            self.assertEqual(variances, original.running_var)
        self.assertEqual(restored(self.inputs, train=False).tolist(), source(self.inputs, train=False).tolist())

        before = [parameter.values[:] for parameter in restored.parameters()]
        nn.MSE()(restored(self.inputs), self.targets).backward()
        optimizer.step()
        self.assertTrue(any(parameter.values != values for parameter, values in zip(restored.parameters(), before, strict=True)))
        self.assertNotEqual(current_norms[0].running_mean, source_norms[0].running_mean)

    def test_identity_shortcut_and_parameterless_layers_roundtrip(self):
        source = nn.Model([nn.Residual(nn.Model([nn.ReLU()])), nn.GlobalAvgPool2d()])
        inputs = [[[[-2, 1], [3, -4]]]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model().load(path)
        self.assertIsNone(restored.layers[0].shortcut)
        self.assertEqual(restored.parameters(), ())
        self.assertEqual(restored(inputs, train=False).tolist(), [[0.5]])


if __name__ == "__main__":
    unittest.main()
