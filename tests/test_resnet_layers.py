"""BatchNorm、残差分支和全局平均池化的数值、梯度与模式。"""

import unittest
from math import sqrt

import qutai as qt
from qutai import Tensor, nn
from qutai.core.context import set_recording


class BatchNormTests(unittest.TestCase):
    def test_training_values_statistics_parameters_and_optimizer(self):
        layer = nn.BatchNorm2d(2, eps=1, momentum=0.25)
        self.assertEqual(layer.weight.values, [1, 1])
        self.assertEqual(layer.bias.values, [0, 0])
        self.assertEqual(layer.running_mean, [0, 0])
        self.assertEqual(layer.running_var, [1, 1])
        layer.weight.values = [2, -0.5]
        layer.bias.values = [1, 3]
        inputs = Tensor([[[[1, 3]], [[-2, 2]]], [[[5, 7]], [[6, 10]]]], requires_grad=True)
        output = layer(inputs)
        self.assertEqual(output.shape, inputs.shape)
        expected = [1 - 6 / sqrt(6), 1 - 2 / sqrt(6), 3 + 3 / sqrt(21), 3 + 1 / sqrt(21),
                    1 + 2 / sqrt(6), 1 + 6 / sqrt(6), 3 - 1 / sqrt(21), 3 - 3 / sqrt(21)]
        for value, reference in zip(output.values, expected, strict=True):
            self.assertAlmostEqual(value, reference)
        self.assertEqual(layer.running_mean, [1, 1])
        for value, reference in zip(layer.running_var, (29 / 12, 89 / 12), strict=True):
            self.assertAlmostEqual(value, reference)

        output.backward([[[[1, -2]], [[3, 0.5]]], [[[-0.7, 1.2]], [[-1, 2]]]])
        for value, reference in zip(layer.weight.grad.values, (1.9 / sqrt(6), -9 / sqrt(21)), strict=True):
            self.assertAlmostEqual(value, reference)
        for value, reference in zip(layer.bias.grad.values, (-0.5, 4.5), strict=True):
            self.assertAlmostEqual(value, reference)
        self.assertIsNotNone(inputs.grad)
        self.assertEqual(layer.parameters(), (layer.weight, layer.bias))
        optimizer = nn.SGD(layer, learning_rate=0.1)
        self.assertEqual(optimizer.parameters, layer.parameters())
        before = layer.running_mean[:], layer.running_var[:]
        optimizer.step()
        self.assertEqual((layer.running_mean, layer.running_var), before)
        self.assertNotEqual(layer.weight.values, [2, -0.5])

    def test_evaluation_uses_running_statistics_and_freezes_them(self):
        layer = nn.BatchNorm2d(2, eps=1)
        layer.weight.values = [2, -3]
        layer.bias.values = [1, 4]
        layer.running_mean[:] = [3, -2]
        layer.running_var[:] = [3, 8]
        inputs = Tensor([[[[1, 5]], [[-5, 1]]]], requires_grad=True)
        output = nn.Model([layer])(inputs, train=False)
        self.assertEqual(output.tolist(), [[[[-1, 3]], [[7, 1]]]])
        self.assertEqual(layer.running_mean, [3, -2])
        self.assertEqual(layer.running_var, [3, 8])
        self.assertFalse(output.requires_grad)
        self.assertEqual(output.parents, ())
        self.assertIsNone(inputs.grad)

    def test_training_statistics_do_not_depend_on_gradient_recording(self):
        layer = nn.BatchNorm2d(1, momentum=0.5)
        with set_recording(False):
            output = nn.Model([layer])([[[[1, 3]]]])
        self.assertFalse(output.requires_grad)
        self.assertEqual(layer.running_mean, [1])
        self.assertEqual(layer.running_var, [1.5])
        self.assertTrue(layer([[[[1, 3]]]]).requires_grad)
        self.assertEqual(layer.running_mean, [1.5])
        layer.weight.requires_grad = layer.bias.requires_grad = False
        self.assertFalse(layer([[[[1, 3]]]]).requires_grad)
        self.assertEqual(layer.running_mean, [1.75])

    def test_outer_prediction_mode_reaches_nested_residual_models(self):
        first, second = nn.BatchNorm2d(1), nn.BatchNorm2d(1)
        block = nn.Residual(nn.Model([first]), shortcut=nn.Model([second]))
        model = nn.Model([nn.Model([block])])
        inputs = Tensor([[[[1, 3]]]], requires_grad=True)
        prediction = model(inputs, train=False)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        for layer in (first, second):
            self.assertEqual(layer.running_mean, [0])
            self.assertEqual(layer.running_var, [1])
        self.assertTrue(model(inputs).requires_grad)
        for layer in (first, second):
            self.assertAlmostEqual(layer.running_mean[0], 0.2)
            self.assertAlmostEqual(layer.running_var[0], 1.1)

    def test_prediction_exception_restores_training_mode(self):
        layer = nn.BatchNorm2d(1, momentum=0.5)
        with self.assertRaises(ValueError):
            nn.Model([layer])([[1, 3]], train=False)
        layer([[[[1, 3]]]])
        self.assertEqual(layer.running_mean, [1])


class ResidualTests(unittest.TestCase):
    def test_identity_shortcut_keeps_negative_values_and_accumulates_gradients(self):
        layer = nn.Linear(2, 2, seed=42)
        layer.weight.values = [1, 0, 0, 2]
        layer.bias.values = [0.5, -0.5]
        block = nn.Residual(nn.Model([layer]))
        inputs = Tensor([[-2, 1]], requires_grad=True)
        result = block(inputs)
        self.assertEqual(result.tolist(), [[-3.5, 2.5]])
        result.backward([[2, -3]])
        self.assertEqual(inputs.grad.tolist(), [[4, -9]])
        self.assertEqual(layer.weight.grad.tolist(), [[-4, 6], [2, -3]])
        self.assertEqual(layer.bias.grad.tolist(), [2, -3])
        self.assertEqual(block.parameters(), layer.parameters())

        shared = Tensor([[-2, 3]], requires_grad=True)
        nn.Residual(nn.Model())(shared).backward([[2, -1]])
        self.assertEqual(shared.grad.tolist(), [[4, -2]])

    def test_projection_shortcut_and_shared_weights(self):
        branch = nn.Conv2d(1, 2, 1, stride=2, seed=42)
        shortcut = nn.Conv2d(1, 2, 1, stride=2, seed=43)
        branch.weight.values = [2, -1]
        branch.bias.values = [0.5, -0.5]
        shortcut.weight = branch.weight
        shortcut.bias.values = [-0.5, 0.5]
        block = nn.Residual(branch, shortcut=shortcut)
        inputs = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], requires_grad=True)
        output = block(inputs)
        self.assertEqual(output.shape, (1, 2, 1, 2))
        self.assertEqual(output.tolist(), [[[[4, 12]], [[-2, -6]]]])
        output.backward([[[[1, 2]], [[3, 5]]]])
        self.assertEqual(inputs.grad.tolist(), [[[[-2, 0, -2, 0], [0, 0, 0, 0]]]])
        self.assertEqual(branch.weight.grad.values, [14, 36])
        self.assertEqual(branch.bias.grad.values, [3, 8])
        self.assertEqual(shortcut.bias.grad.values, [3, 8])
        parameters = (branch.weight, branch.bias, shortcut.bias)
        self.assertEqual(block.parameters(), parameters)
        self.assertEqual(nn.Model([block, branch]).parameters(), parameters)


class GlobalAveragePoolTests(unittest.TestCase):
    def test_batch_channel_means_and_weighted_gradients(self):
        layer = nn.GlobalAvgPool2d()
        inputs = Tensor(list(range(1, 25)), shape=(2, 2, 2, 3), requires_grad=True)
        result = layer(inputs)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[3.5, 9.5], [15.5, 21.5]])
        result.backward([[2, -3], [4, 0.6]])
        for value, reference in zip(inputs.grad.values, [seed / 6 for seed in (2, -3, 4, 0.6) for _ in range(6)], strict=True):
            self.assertAlmostEqual(value, reference)
        self.assertEqual(layer.parameters(), ())
        self.assertEqual(layer(inputs.tolist()).tolist(), result.tolist())


if __name__ == "__main__":
    unittest.main()
