"""模型组合、共享参数和训练/预测模式。"""

import unittest

import qutai as qt
from qutai import Tensor, nn
from qutai.core.context import set_recording


class ModelTests(unittest.TestCase):
    def test_composition_and_gradients(self):
        first = nn.Linear(2, 2, seed=42)
        first.weight.values = [1, 0, 0, 1]
        first.bias.values = [0, 0]
        last = nn.Linear(2, 1, seed=43)
        last.weight.values = [2, 3]
        last.bias.values = [0.5]
        model = nn.Model([first, nn.ReLU(), last])
        inputs = Tensor([[-1, 2]], requires_grad=True)
        result = model(inputs)
        self.assertEqual(result.tolist(), [[6.5]])
        result.backward()
        self.assertEqual(inputs.grad.tolist(), [[0, 3]])
        self.assertEqual(first.weight.grad.tolist(), [[0, -3], [0, 6]])
        self.assertEqual(first.bias.grad.tolist(), [0, 3])
        self.assertEqual(last.weight.grad.tolist(), [[0], [2]])
        self.assertEqual(last.bias.grad.tolist(), [1])

    def test_nested_models_and_shared_parameters(self):
        shared = nn.Linear(2, 2, seed=42)
        last = nn.Linear(2, 1, seed=43)
        inner = nn.Model([nn.ReLU(), shared, last])
        model = nn.Model(layer for layer in (shared, inner))
        expected = (shared.weight, shared.bias, last.weight, last.bias)
        self.assertEqual(model.parameters(), expected)
        self.assertEqual(model.parameters(), expected)
        self.assertEqual(model([[1, 2]]).shape, (1, 1))
        self.assertFalse(model([[1, 2]], train=False).requires_grad)

    def test_prediction_preserves_parameters_gradients_and_input(self):
        model = nn.Model([nn.Linear(2, 3, seed=42), nn.ReLU(), nn.Linear(3, 1, seed=43)])
        inputs = Tensor([[1, 2]], requires_grad=True)
        training = model(inputs, train=True)
        nn.MSE()(training, [[0]]).backward()
        parameters = model.parameters()
        values = [parameter.values[:] for parameter in parameters]
        gradients = [parameter.grad for parameter in parameters]
        input_gradient = inputs.grad
        prediction = model(inputs, train=False)
        self.assertEqual(prediction.tolist(), training.tolist())
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertIsNone(prediction.backward_rule)
        self.assertIsNone(prediction.grad)
        self.assertTrue(inputs.requires_grad)
        self.assertIs(inputs.grad, input_gradient)
        for parameter, before, gradient in zip(parameters, values, gradients, strict=True):
            self.assertEqual(parameter.values, before)
            self.assertTrue(parameter.requires_grad)
            self.assertIs(parameter.grad, gradient)
        optimizer = nn.SGD(parameters)
        optimizer.zero_grad()
        nn.MSE()(model(inputs), [[0]]).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_prediction_exception_restores_recording(self):
        model = nn.Model([nn.Linear(2, 1, seed=42)])
        with self.assertRaises(ValueError):
            model([1], train=False)
        result = model([[1, 2]])
        self.assertTrue(result.requires_grad)
        result.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_outer_disabled_mode_cannot_be_reenabled(self):
        inputs = Tensor(2, requires_grad=True)
        with set_recording(False):
            with set_recording(True):
                nested = qt.square(inputs)
            following = qt.add(inputs, 1)
        self.assertFalse(nested.requires_grad)
        self.assertEqual(nested.parents, ())
        self.assertFalse(following.requires_grad)
        qt.square(inputs).backward()
        self.assertEqual(inputs.grad.item(), 4)

    def test_empty_model_identity_and_prediction(self):
        inputs = qt.square(Tensor([1, 2], requires_grad=True))
        model = nn.Model([])
        self.assertEqual(model.parameters(), ())
        self.assertIs(model(inputs), inputs)
        prediction = model(inputs, train=False)
        self.assertIsNot(prediction, inputs)
        self.assertEqual(prediction.tolist(), inputs.tolist())
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertTrue(inputs.requires_grad)
        self.assertTrue(inputs.parents)
        self.assertIsInstance(model([1, 2]), Tensor)


if __name__ == "__main__":
    unittest.main()
