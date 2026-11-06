"""层的并联相加、ResNetBlock 与原组合等价，以及模型读写。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import qutai as qt
from qutai import Tensor, nn


class Scale(nn.Layer):
    def __init__(self, factor):
        self.factor, self.inputs = factor, []

    def __call__(self, data):
        self.inputs.append(data)
        return qt.multiply(data, self.factor)


class Transpose(nn.Layer):
    def __call__(self, data):
        return qt.transpose(data)


class CustomLayer:
    """已有用户层无需为了交给优化器而继承 Layer。"""
    def __init__(self):
        self.weight = Tensor(2, requires_grad=True)

    def __call__(self, data):
        return qt.multiply(data, self.weight)

    def parameters(self):
        return (self.weight,)


def old_block(in_channels, out_channels, stride, seed):
    """Notebook 原来的 make_block，连同其后单独添加的 ReLU。"""
    branch = nn.Model([
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, seed=seed),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, seed=seed + 1),
        nn.BatchNorm2d(out_channels),
    ])
    shortcut = None
    if stride != 1 or in_channels != out_channels:
        shortcut = nn.Model([nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False, seed=seed + 2),
                             nn.BatchNorm2d(out_channels)])
    return nn.Model([nn.Residual(branch, shortcut), nn.ReLU()])


def normalizations(layer):
    if isinstance(layer, nn.BatchNorm2d): return (layer,)
    if isinstance(layer, nn.Residual):
        children = (layer.branch,) if layer.shortcut is None else (layer.branch, layer.shortcut)
    else:
        children = getattr(layer, "layers", ())
    return tuple(norm for child in children for norm in normalizations(child))


class LayerCompositionTests(unittest.TestCase):
    def test_base_defaults_and_all_builtin_layers_can_be_added(self):
        self.assertEqual(nn.Layer().parameters(), ())
        with self.assertRaisesRegex(NotImplementedError, "[\u4e00-\u9fff]"):
            nn.Layer()(Tensor(1))
        layers = (nn.Linear(1, 1), nn.Conv2d(1, 1, 1), nn.MaxPool2d(2), nn.Flatten(),
                  nn.BatchNorm2d(1), nn.GlobalAvgPool2d(), nn.ReLU(), nn.Softmax(),
                  nn.Identity(), nn.Model(), nn.Residual(nn.Identity()), nn.ResNetBlock(1, 1, seed=100))
        for layer in layers:
            with self.subTest(layer=type(layer).__name__):
                self.assertIsInstance(layer, nn.Layer)
                self.assertIsInstance(layer + nn.Identity(), nn.Residual)

    def test_construction_is_lazy_and_three_branches_receive_the_same_input(self):
        branches = [Scale(factor) for factor in (1, 2, 3)]
        combined = branches[0] + branches[1] + branches[2]
        self.assertTrue(all(branch.inputs == [] for branch in branches))
        inputs = Tensor([-2, 3], requires_grad=True)
        output = combined(inputs)
        self.assertEqual(output.tolist(), [-12, 18])
        for branch in branches:
            self.assertEqual(len(branch.inputs), 1)
            self.assertIs(branch.inputs[0], inputs)
        qt.sum(output).backward()
        self.assertEqual(inputs.grad.tolist(), [6, 6])
        self.assertEqual(combined.parameters(), ())

    def test_model_plus_layer_collects_shared_parameters_once(self):
        shared = nn.Linear(1, 1, seed=42)
        shared.weight.values, shared.bias.values = [2], [1]
        combined = nn.Model([shared]) + shared + shared
        inputs = Tensor([[4]], requires_grad=True)
        result = combined(inputs)
        self.assertEqual(result.tolist(), [[27]])
        result.backward()
        self.assertEqual(inputs.grad.tolist(), [[6]])
        self.assertEqual(shared.weight.grad.values, [12])
        self.assertEqual(shared.bias.grad.values, [3])
        self.assertEqual(combined.parameters(), shared.parameters())
        nn.SGD(combined, learning_rate=0.1).step()
        self.assertAlmostEqual(shared.weight.item(), 0.8)
        self.assertAlmostEqual(shared.bias.item(), 0.7)

    def test_addition_requires_identical_shapes_even_when_broadcastable(self):
        for combined, inputs in ((nn.Linear(2, 1, seed=42) + nn.Identity(), [[1, 2], [3, 4]]),
                                 (nn.Identity() + Transpose(), [[1, 2]])):
            with self.subTest(inputs=inputs):
                with self.assertRaisesRegex(ValueError, "形状"):
                    combined(inputs)

    def test_existing_custom_parameter_source_remains_usable_by_sgd(self):
        layer = CustomLayer()
        self.assertNotIsInstance(layer, nn.Layer)
        optimizer = nn.SGD(layer, learning_rate=0.1)
        layer(3).backward()
        optimizer.step()
        self.assertEqual(optimizer.parameters, (layer.weight,))
        self.assertAlmostEqual(layer.weight.item(), 1.7)

    def test_resnet_block_matches_old_identity_and_projection_blocks(self):
        values = [(index - 12) / 10 for index in range(24)]
        for out_channels, stride in ((1, 1), (2, 2)):
            with self.subTest(out_channels=out_channels, stride=stride):
                block = nn.ResNetBlock(1, out_channels, stride=stride, seed=100)
                previous = old_block(1, out_channels, stride, 100)
                inputs = Tensor(values, shape=(2, 1, 3, 4), requires_grad=True)
                old_inputs = Tensor(values, shape=inputs.shape, requires_grad=True)
                for current, original in zip(block.parameters(), previous.parameters(), strict=True):
                    self.assertEqual(current.shape, original.shape)
                    self.assertEqual(current.values, original.values)
                output, old_output = block(inputs), previous(old_inputs)
                self.assertEqual(output.shape, old_output.shape)
                self.assertEqual(output.values, old_output.values)
                seed = Tensor([(index % 7 - 3) / 5 for index in range(output.size)], shape=output.shape)
                output.backward(seed)
                old_output.backward(seed)
                self.assertEqual(inputs.grad.values, old_inputs.grad.values)
                for current, original in zip(block.parameters(), previous.parameters(), strict=True):
                    self.assertEqual(current.grad.values, original.grad.values)
                nn.SGD(block, learning_rate=0.01, momentum=0.9, weight_decay=0.001).step()
                nn.SGD(previous, learning_rate=0.01, momentum=0.9, weight_decay=0.001).step()
                for current, original in zip(block.parameters(), previous.parameters(), strict=True):
                    self.assertEqual(current.values, original.values)
                norms = normalizations(block)
                before = [(norm.running_mean[:], norm.running_var[:]) for norm in norms]
                for current, original in zip(norms, normalizations(previous), strict=True):
                    self.assertEqual((current.running_mean, current.running_var), (original.running_mean, original.running_var))
                prediction = nn.Model([block])(inputs, train=False)
                self.assertEqual(prediction.values, previous(old_inputs, train=False).values)
                self.assertFalse(prediction.requires_grad)
                self.assertEqual([(norm.running_mean, norm.running_var) for norm in norms], before)

    def test_sum_of_blocks_roundtrips_shared_parameters_and_running_state(self):
        source_block = nn.ResNetBlock(1, 1, seed=100)
        source = nn.Model([source_block + source_block, nn.GlobalAvgPool2d()])
        inputs = Tensor([(index - 12) / 10 for index in range(24)], shape=(2, 1, 3, 4))
        source(inputs)
        target_block = nn.ResNetBlock(1, 1, seed=7)
        target = nn.Model([target_block + target_block, nn.GlobalAvgPool2d()])
        target(inputs)
        optimizer = nn.SGD(target)
        target_norms = normalizations(target)
        buffers = [(norm.running_mean, norm.running_var) for norm in target_norms]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            self.assertEqual(len(json.loads(path.read_text())["parameters"]), len(source_block.parameters()))
            restored = nn.Model().load(path)
            self.assertIs(target.load(path), target)
        left, right = restored.layers[0].branch, restored.layers[0].shortcut
        self.assertIsInstance(left, nn.ResNetBlock)
        self.assertIsInstance(right, nn.ResNetBlock)
        self.assertIsInstance(left.layers[0].shortcut, nn.Identity)
        for first, second in zip(left.parameters(), right.parameters(), strict=True):
            self.assertIs(first, second)
        loaded_norms = normalizations(restored)
        self.assertIs(loaded_norms[0], loaded_norms[2])
        self.assertIs(loaded_norms[1], loaded_norms[3])
        for loaded, original in zip(loaded_norms, normalizations(source), strict=True):
            self.assertEqual((loaded.running_mean, loaded.running_var), (original.running_mean, original.running_var))
        for current, reference in zip(target.parameters(), optimizer.parameters, strict=True):
            self.assertIs(current, reference)
        for current, previous in zip(normalizations(target), target_norms, strict=True):
            self.assertIs(current, previous)
        for norm, (means, variances), original in zip(normalizations(target), buffers, normalizations(source), strict=True):
            self.assertIs(norm.running_mean, means)
            self.assertIs(norm.running_var, variances)
            self.assertEqual(means, original.running_mean)
            self.assertEqual(variances, original.running_var)
        expected = source(inputs, train=False).values
        self.assertEqual(restored(inputs, train=False).values, expected)
        self.assertEqual(target(inputs, train=False).values, expected)

    def test_old_residual_archive_still_loads(self):
        saved = {"model": {"type": "Model", "layers": [{"type": "Residual",
                  "branch": {"type": "Model", "layers": [{"type": "ReLU"}]}, "shortcut": None}]}, "parameters": []}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "old.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            model = nn.Model().load(path)
        self.assertIsInstance(model.layers[0], nn.Residual)
        self.assertIsNone(model.layers[0].shortcut)
        self.assertEqual(model([[-2, 3]], train=False).tolist(), [[-2, 6]])


if __name__ == "__main__":
    unittest.main()
