"""优化器更新、参数共享、迭代输入和梯度生命周期。"""

import unittest

import qutai as qt
from qutai import Tensor, nn


class SGDTests(unittest.TestCase):
    def test_scalar_steps_with_fresh_gradients(self):
        parameter = Tensor(3, requires_grad=True)
        optimizer = nn.SGD([parameter], learning_rate=0.1)
        qt.square(parameter).backward()
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 2.4)
        self.assertEqual(parameter.grad.item(), 6)
        self.assertTrue(parameter.requires_grad)
        self.assertEqual(parameter.parents, ())
        self.assertIs(optimizer.parameters[0], parameter)

        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)
        qt.square(parameter).backward()
        self.assertAlmostEqual(parameter.grad.item(), 4.8)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.92)

    def test_matrix_and_broadcast_bias(self):
        weight = Tensor([[1, 2], [3, 4]], requires_grad=True)
        bias = Tensor([0.5, -0.5], requires_grad=True)
        optimizer = nn.SGD([weight, bias], learning_rate=0.25)
        qt.sum(qt.add(weight, bias)).backward()
        optimizer.step()
        self.assertEqual(weight.tolist(), [[0.75, 1.75], [2.75, 3.75]])
        self.assertEqual(bias.tolist(), [0, -1])
        self.assertEqual(weight.shape, (2, 2))
        self.assertEqual(bias.shape, (2,))

    def test_missing_gradient_is_skipped(self):
        used = Tensor(1, requires_grad=True)
        unused = Tensor(4, requires_grad=True)
        optimizer = nn.SGD([unused, used], learning_rate=0.1)
        optimizer.zero_grad()
        qt.square(used).backward()
        optimizer.step()
        self.assertAlmostEqual(used.item(), 0.8)
        self.assertEqual(unused.item(), 4)
        self.assertIsNone(unused.grad)
        optimizer.zero_grad()
        self.assertIsNone(used.grad)

    def test_generator_and_shared_parameter(self):
        parameter = Tensor(1, requires_grad=True)
        optimizer = nn.SGD((p for p in (parameter, parameter)), learning_rate=0.25)
        qt.add(parameter, parameter).backward()
        optimizer.step()
        self.assertEqual(parameter.item(), 0.5)
        self.assertEqual(len(optimizer.parameters), 1)
        optimizer.step()
        self.assertEqual(parameter.item(), 0)
        self.assertEqual(parameter.grad.item(), 2)
        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)

    def test_linear_layer_can_be_optimized_directly(self):
        layer = nn.Linear(2, 1, seed=42)
        layer.weight.values = [1, 2]
        layer.bias.values = [0.5]
        optimizer = nn.SGD(layer, learning_rate=0.25)

        qt.sum(layer([[1, 3]])).backward()
        optimizer.step()
        self.assertEqual(layer.weight.tolist(), [[0.75], [1.25]])
        self.assertEqual(layer.bias.tolist(), [0.25])
        optimizer.zero_grad()
        self.assertIsNone(layer.weight.grad)
        self.assertIsNone(layer.bias.grad)

    def test_nested_model_updates_shared_parameters_once_and_ignores_other_attributes(self):
        shared = nn.Linear(1, 1, seed=42)
        shared.weight.values = [2]
        shared.bias.values = [1]
        model = nn.Model([shared, nn.Model([nn.ReLU(), shared])])
        model.unrelated = Tensor(5, requires_grad=True)
        other = nn.Model([nn.Linear(1, 1, seed=7)])
        other_values = [parameter.values[:] for parameter in other.parameters()]
        qt.square(model.unrelated).backward()
        qt.sum(other([[1]])).backward()
        other_gradients = [parameter.grad for parameter in other.parameters()]
        optimizer = nn.SGD(model, learning_rate=0.1)

        optimizer.zero_grad()
        self.assertEqual(model.unrelated.grad.item(), 10)
        qt.sum(model([[3]])).backward()
        optimizer.step()
        self.assertAlmostEqual(shared.weight.item(), 0.7)
        self.assertAlmostEqual(shared.bias.item(), 0.7)
        self.assertEqual(model.unrelated.item(), 5)
        self.assertEqual([parameter.values for parameter in other.parameters()], other_values)
        optimizer.zero_grad()
        self.assertIsNone(shared.weight.grad)
        self.assertIsNone(shared.bias.grad)
        self.assertEqual(model.unrelated.grad.item(), 10)
        for parameter, gradient in zip(other.parameters(), other_gradients, strict=True):
            self.assertIs(parameter.grad, gradient)


class MomentumTests(unittest.TestCase):
    def test_momentum_and_weight_decay_hand_calculation(self):
        parameter = Tensor(2, requires_grad=True)
        optimizer = nn.SGD([parameter], learning_rate=0.1, momentum=0.9, weight_decay=0.1)
        parameter.grad = Tensor(3)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.68)
        self.assertAlmostEqual(optimizer.velocities[parameter][0], 3.2)
        self.assertEqual(parameter.grad.item(), 3)
        parameter.grad = Tensor(1)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.2752)
        self.assertAlmostEqual(optimizer.velocities[parameter][0], 4.048)

    def test_missing_gradient_keeps_momentum_and_shared_parameter_updates_once(self):
        parameter = Tensor(2, requires_grad=True)
        optimizer = nn.SGD([parameter, parameter], learning_rate=0.1, momentum=0.5)
        parameter.grad = Tensor(1)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.9)
        optimizer.zero_grad()
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.9)
        self.assertEqual(optimizer.velocities[parameter], [1])
        parameter.grad = Tensor(2)
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 1.65)


if __name__ == "__main__":
    unittest.main()
