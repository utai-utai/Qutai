"""验证回归 Notebook 的训练结果和独立数据预测。"""

import unittest
from random import Random

from qutai import nn
from tests.notebook_helpers import run_notebook


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linear_model = run_notebook("regression", stop_after="linear-results")["model"]
        cls.nonlinear_model = run_notebook("regression")["model"]

    def test_multilayer_network_learns_nonlinear_relation(self):
        model = self.nonlinear_model
        loss_function = nn.MSE()
        inputs = [[-1 + 2 * i / 64] for i in range(65)]
        targets = [[x * x] for [x] in inputs]
        untrained = nn.Model([nn.Linear(1, 16, seed=42), nn.ReLU(), nn.Linear(16, 1, seed=43)])
        initial = loss_function(untrained(inputs, train=False), targets).item()
        prediction = model(inputs, train=False)
        final = loss_function(prediction, targets).item()
        self.assertLess(final, initial * 0.01)

        rng = Random(11)
        held_out = [[rng.uniform(-1, 1)] for _ in range(64)]
        expected = [[x * x] for [x] in held_out]
        self.assertLess(loss_function(model(held_out, train=False), expected).item(), 0.003)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertEqual(len(model.parameters()), 4)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.parents == () for parameter in model.parameters()))

    def test_linear_regression_converges_and_generalizes(self):
        model = self.linear_model
        weight, bias = model.parameters()
        self.assertAlmostEqual(weight.values[0], 2, places=5)
        self.assertAlmostEqual(weight.values[1], -3, places=5)
        self.assertAlmostEqual(bias.item(), 0.5, places=5)

        rng = Random(7)
        held_out = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(16)]
        expected = [[2 * a - 3 * b + 0.5] for a, b in held_out]
        prediction = model(held_out, train=False)
        self.assertLess(nn.MSE()(prediction, expected).item(), 1e-8)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertTrue(weight.requires_grad)
        self.assertTrue(bias.requires_grad)
        self.assertEqual(weight.parents, ())
        self.assertEqual(bias.parents, ())


if __name__ == "__main__":
    unittest.main()
