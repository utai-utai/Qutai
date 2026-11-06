"""批归一化状态加载失败的原子性，以及预测统计与求导的独立性。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import qutai as qt
from qutai import Tensor, nn
from qutai.core.context import set_training


def make_model() -> nn.Model:
    return nn.Model([nn.BatchNorm2d(1, momentum=0.3), nn.Model([nn.ReLU(), nn.BatchNorm2d(1, momentum=0.4)])])


class ResNetStateEdgeTests(unittest.TestCase):
    def test_late_invalid_statistics_or_parameters_preserve_existing_state(self):
        source, restored = make_model(), make_model()
        source.layers[0].weight.values = [2]
        source.layers[1].layers[1].bias.values = [3]
        inputs = [[[[1, 3]]], [[[5, 7]]]]
        source(inputs)
        qt.sum(restored(inputs)).backward()
        parameters = restored.parameters()
        values = [parameter.values[:] for parameter in parameters]
        gradients = [parameter.grad for parameter in parameters]
        norms = (restored.layers[0], restored.layers[1].layers[1])
        buffers = [(layer.running_mean, layer.running_var) for layer in norms]
        statistics = [(average[:], variance[:]) for average, variance in buffers]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = path.read_text(encoding="utf-8")
            for issue in ("statistics", "parameter"):
                with self.subTest(issue=issue):
                    data = json.loads(saved)
                    if issue == "statistics":
                        data["model"]["layers"][1]["layers"][1]["running_var"] = [-1]
                    else:
                        data["parameters"][-1]["values"] = ["invalid"]
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        restored.load(path)
                    self.assertEqual(restored.parameters(), parameters)
                    for parameter, previous, gradient in zip(parameters, values, gradients, strict=True):
                        self.assertEqual(parameter.values, previous)
                        self.assertIs(parameter.grad, gradient)
                    for layer, (average, variance), previous in zip(norms, buffers, statistics, strict=True):
                        self.assertIs(layer.running_mean, average)
                        self.assertIs(layer.running_var, variance)
                        self.assertEqual((average, variance), previous)

    def test_outer_evaluation_statistics_allow_gradients_through_nested_models(self):
        layer = nn.BatchNorm2d(1, eps=3)
        layer.weight.values, layer.bias.values = [2], [1]
        layer.running_mean[:], layer.running_var[:] = [2], [1]
        inputs = Tensor([1, 3], shape=(2, 1, 1, 1), requires_grad=True)
        model = nn.Model([nn.Model([layer])])
        with set_training(False):
            result = model(inputs)
        self.assertTrue(result.requires_grad)
        self.assertEqual(result.values, [0, 2])
        self.assertEqual(layer.running_mean, [2])
        self.assertEqual(layer.running_var, [1])
        result.backward(Tensor([3, -1], shape=result.shape))
        self.assertEqual(inputs.grad.values, [3, -1])
        self.assertEqual(layer.weight.grad.values, [-2])
        self.assertEqual(layer.bias.grad.values, [2])
        layer(inputs)
        self.assertEqual(layer.running_mean, [2])
        self.assertAlmostEqual(layer.running_var[0], 1.1)

    def test_shared_batch_norm_roundtrip_keeps_state_and_continued_training(self):
        shared = nn.BatchNorm2d(1, momentum=0.5)
        source = nn.Model([shared, nn.Model([nn.ReLU(), shared])])
        inputs = [[[[1, 3]]]]
        source(inputs)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model().load(path)
        first, second = restored.layers[0], restored.layers[1].layers[1]
        self.assertIs(first, second)
        self.assertIs(first.running_mean, second.running_mean)
        self.assertIs(first.running_var, second.running_var)
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

        next_inputs, targets = [[[[10, 20]]]], [[[[0.2, -0.3]]]]
        for model in (source, restored):
            optimizer = nn.SGD(model, learning_rate=0.01)
            optimizer.zero_grad()
            nn.MSE()(model(next_inputs), targets).backward()
            optimizer.step()
        self.assertEqual(first.running_mean, shared.running_mean)
        self.assertEqual(first.running_var, shared.running_var)
        self.assertEqual([parameter.values for parameter in restored.parameters()], [parameter.values for parameter in source.parameters()])
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_distinct_batch_norms_keep_independent_statistics_with_shared_parameters(self):
        first, second = nn.BatchNorm2d(1), nn.BatchNorm2d(1)
        second.weight, second.bias = first.weight, first.bias
        first.running_mean[:], first.running_var[:] = [-2], [0.5]
        second.running_mean[:], second.running_var[:] = [4], [3]
        source = nn.Model([first, nn.ReLU(), second])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved), {"model", "parameters"})
            restored = nn.Model().load(path)
            shared = nn.BatchNorm2d(1)
            incompatible = nn.Model([shared, nn.ReLU(), shared])
            with self.assertRaisesRegex(ValueError, "结构"):
                incompatible.load(path)
        left, right = restored.layers[0], restored.layers[2]
        self.assertIsNot(left, right)
        self.assertIs(left.weight, right.weight)
        self.assertIs(left.bias, right.bias)
        self.assertIsNot(left.running_mean, right.running_mean)
        self.assertIsNot(left.running_var, right.running_var)
        self.assertEqual((left.running_mean, left.running_var), ([-2], [0.5]))
        self.assertEqual((right.running_mean, right.running_var), ([4], [3]))
        inputs = [[[[1, 3]]]]
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())


if __name__ == "__main__":
    unittest.main()
