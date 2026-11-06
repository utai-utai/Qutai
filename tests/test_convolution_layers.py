"""卷积、池化、展平层的组合训练与模型保存。"""

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import qutai as qt
from qutai import Tensor, nn


def make_model(seed: int) -> nn.Model:
    return nn.Model([
        nn.Conv2d(1, 2, (2, 3), stride=(1, 2), padding=(1, 0), bias=False, seed=seed),
        nn.ReLU(),
        nn.MaxPool2d((2, 1), stride=(2, 1)),
        nn.Flatten(),
        nn.Linear(8, 3, seed=seed + 1),
    ])


class ConvolutionLayerTests(unittest.TestCase):
    def test_known_convolution_and_gradients(self):
        layer = nn.Conv2d(1, 1, 2, seed=42)
        layer.weight.values = [1, 2, 3, 4]
        layer.bias.values = [0.5]
        data = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        inputs = Tensor(data, requires_grad=True)
        result = layer(inputs)
        self.assertEqual(result.tolist(), [[[[37.5, 47.5], [67.5, 77.5]]]])
        self.assertIsInstance(layer(data), Tensor)
        qt.sum(result).backward()
        self.assertEqual(inputs.grad.tolist(), [[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]])
        self.assertEqual(layer.weight.grad.values, [12, 16, 24, 28])
        self.assertEqual(layer.bias.grad.values, [4])

    def test_initialization_and_parameter_ownership(self):
        state = random.getstate()
        first = nn.Conv2d(2, 3, (3, 2), stride=2, padding=1, bias=False, seed=42)
        second = nn.Conv2d(2, 3, (3, 2), seed=42)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first.weight.shape, (3, 2, 3, 2))
        self.assertEqual(first.weight.values, second.weight.values)
        self.assertIsNot(first.weight, second.weight)
        self.assertEqual(first.parameters(), (first.weight,))
        self.assertIsNone(first.bias)
        self.assertTrue(first.weight.requires_grad)
        bound = (6 / 12) ** 0.5
        self.assertTrue(all(-bound <= value <= bound for value in first.weight.values))
        self.assertEqual(first.stride, (2, 2))
        self.assertEqual(first.padding, (1, 1))

    def test_convolution_pooling_flatten_and_optimizer(self):
        layer = nn.Conv2d(1, 1, 2, bias=False, seed=42)
        layer.weight.values = [1, 1, 1, 1]
        model = nn.Model([layer, nn.MaxPool2d(2), nn.Flatten()])
        inputs = Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True)
        result = model(inputs)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.tolist(), [[28]])
        optimizer = nn.SGD(model, learning_rate=0.001)
        qt.sum(result).backward()
        self.assertEqual(inputs.grad.tolist(), [[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]])
        self.assertEqual(layer.weight.grad.values, [5, 6, 8, 9])
        optimizer.step()
        self.assertLess(model(inputs, train=False).item(), 28)
        optimizer.zero_grad()
        self.assertIsNone(layer.weight.grad)
        self.assertEqual(len(optimizer.parameters), 1)

    def test_pooling_tuple_options_and_tensor_result(self):
        layer = nn.MaxPool2d((2, 3), stride=(1, 2), padding=(0, 1))
        result = layer([[[[1, 2, 3, 4], [5, 6, 7, 8]]]])
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.tolist(), [[[[6, 8]]]])
        self.assertEqual(layer.parameters(), ())
        self.assertFalse(result.requires_grad)

    def test_flatten_axes_and_gradient_order(self):
        inputs = Tensor(list(range(24)), shape=(2, 3, 2, 2), requires_grad=True)
        result = nn.Flatten(-2)(inputs)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertEqual(result.values, inputs.values)
        result.backward(Tensor(list(range(1, 25)), shape=result.shape))
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertEqual(inputs.grad.values, list(range(1, 25)))
        self.assertEqual(nn.Flatten()([[[1, 2], [3, 4]]]).shape, (1, 4))
        self.assertEqual(nn.Flatten(0)([[1, 2], [3, 4]]).shape, (4,))
        self.assertEqual(nn.Flatten(-1)(3).tolist(), [3])
        self.assertEqual(nn.Flatten().parameters(), ())

    def test_layer_options_reject_invalid_sizes(self):
        for channels, outputs, kernel, stride, padding in ((0, 1, 2, 1, 0), (1, 0, 2, 1, 0), (1, 1, 0, 1, 0), (1, 1, 2, 0, 0), (1, 1, 2, 1, -1)):
            with self.subTest(channels=channels, outputs=outputs, kernel=kernel, stride=stride, padding=padding):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    nn.Conv2d(channels, outputs, kernel, stride=stride, padding=padding)
        for kernel, stride, padding in ((0, None, 0), (2, 0, 0), (2, None, -1)):
            with self.subTest(kernel=kernel, stride=stride, padding=padding):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    nn.MaxPool2d(kernel, stride=stride, padding=padding)
        with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
            nn.Flatten(4)([[1, 2]])


class ConvolutionSerializationTests(unittest.TestCase):
    def test_roundtrip_preserves_config_flags_and_predictions(self):
        source = make_model(42)
        source.layers[0].weight.requires_grad = False
        inputs = Tensor([index / 10 for index in range(40)], shape=(2, 1, 4, 5))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            data = json.loads(path.read_text())
            self.assertEqual(set(data), {"model", "parameters"})
            self.assertEqual([layer["type"] for layer in data["model"]["layers"]], ["Conv2d", "ReLU", "MaxPool2d", "Flatten", "Linear"])
            self.assertEqual(data["model"]["layers"][0]["kernel_size"], [2, 3])
            restored = nn.Model().load(path)
        convolution, _, pooling, flatten, _ = restored.layers
        self.assertEqual(convolution.kernel_size, (2, 3))
        self.assertEqual(convolution.stride, (1, 2))
        self.assertEqual(convolution.padding, (1, 0))
        self.assertIsNone(convolution.bias)
        self.assertFalse(convolution.weight.requires_grad)
        self.assertEqual(pooling.kernel_size, (2, 1))
        self.assertEqual(pooling.stride, (2, 1))
        self.assertEqual(flatten.start_dim, 1)
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_load_into_model_preserves_optimizer_references(self):
        source, restored = make_model(42), make_model(7)
        inputs = Tensor([index / 10 for index in range(40)], shape=(2, 1, 4, 5))
        optimizer = nn.SGD(restored)
        parameters = restored.parameters()
        qt.sum(restored(inputs)).backward()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored.load(path)
        for previous, current in zip(optimizer.parameters, restored.parameters(), strict=True):
            self.assertIs(previous, current)
            self.assertIsNone(current.grad)
        self.assertEqual(parameters, restored.parameters())
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_nested_layers_keep_shared_weights_and_optional_bias(self):
        first = nn.Conv2d(1, 1, 1, bias=False, seed=42)
        last = nn.Conv2d(1, 1, 1, seed=43)
        last.weight = first.weight
        source = nn.Model([first, nn.Model([nn.ReLU(), last]), nn.MaxPool2d(1), nn.Flatten(-1)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            restored = nn.Model().load(path)
        self.assertEqual(len(restored.parameters()), 2)
        self.assertIs(restored.layers[0].weight, restored.layers[1].layers[1].weight)
        self.assertIsNone(restored.layers[0].bias)
        self.assertIsNotNone(restored.layers[1].layers[1].bias)
        self.assertEqual(restored.layers[-1].start_dim, -1)
        inputs = [[[[1, 2], [3, 4]]]]
        self.assertEqual(restored(inputs, train=False).tolist(), source(inputs, train=False).tolist())

    def test_invalid_convolution_metadata_is_rejected(self):
        source = nn.Model([nn.Conv2d(1, 1, 1, bias=False, seed=42)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            source.save(path)
            saved = json.loads(path.read_text())
            for field, value in (("kernel_size", [2, 2]), ("parameters", [-1]), ("parameters", [])):
                with self.subTest(field=field, value=value):
                    data = json.loads(json.dumps(saved))
                    data["model"]["layers"][0][field] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                        nn.Model().load(path)


if __name__ == "__main__":
    unittest.main()
