"""模型与优化器配对恢复，核对两次更新及小语言模型的训练流程。"""

import json
import unittest
from math import isfinite, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, lm, nn
from qutai.nn.optimizers import AdamW, clip_grad_norm


def shared_model() -> nn.Model:
    first, last = nn.Linear(2, 2, seed=7), nn.Linear(2, 2, seed=8)
    first.weight.values = [0.5, -0.25, 0.75, 0.4]
    first.bias.values = [0.1, 0.2]
    last.weight = first.weight
    last.bias.values = [-0.1, 0.3]
    return nn.Model([first, nn.ReLU(), last])


def train_step(model, optimizer: AdamW, loss_function, inputs, targets) -> tuple[float, float]:
    optimizer.zero_grad()
    loss = loss_function(model(inputs), targets)
    loss.backward()
    norm = clip_grad_norm(model, max_norm=0.25)
    optimizer.step()
    return loss.item(), norm


def optimizer_record(optimizer: AdamW, path: Path) -> dict:
    optimizer.save(path)
    return json.loads(path.read_text(encoding="utf-8"))


class TrainingResumeTests(unittest.TestCase):
    def test_adamw_two_steps_match_save_and_resume_with_shared_parameters(self):
        continuous, interrupted = shared_model(), shared_model()
        settings = dict(learning_rate=0.03, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.02)
        optimizer, checkpoint = AdamW(continuous, **settings), AdamW(interrupted, **settings)
        inputs = [[1, 2], [-1, 1], [2, -0.5]]
        targets = [[0.5, -0.5], [0, 1], [-1, 0.5]]
        loss_function = nn.MSE()
        initial = [parameter.values[:] for parameter in continuous.parameters()]

        first = train_step(continuous, optimizer, loss_function, inputs, targets)
        self.assertEqual(first, train_step(interrupted, checkpoint, loss_function, inputs, targets))
        with TemporaryDirectory() as folder:
            model_path, optimizer_path = Path(folder) / "model.json", Path(folder) / "optimizer.json"
            interrupted.save(model_path)
            before = optimizer_record(checkpoint, optimizer_path)
            self.assertEqual(before, optimizer_record(optimizer, Path(folder) / "continuous.json"))

            restored = nn.Model().load(model_path)
            resumed = AdamW(restored, learning_rate=9, betas=(0.1, 0.2), eps=1, weight_decay=0.7)
            self.assertIs(resumed.load(optimizer_path), resumed)
            self.assertEqual(before, optimizer_record(resumed, Path(folder) / "restored.json"))
            self.assertEqual(resumed.learning_rate, settings["learning_rate"])
            self.assertEqual(resumed.betas, settings["betas"])
            self.assertEqual(resumed.eps, settings["eps"])
            self.assertEqual(resumed.weight_decay, settings["weight_decay"])
            self.assertIs(restored.layers[0].weight, restored.layers[2].weight)
            self.assertEqual(len(restored.parameters()), 3)
            for parameter, optimized in zip(restored.parameters(), resumed.parameters, strict=True):
                self.assertIs(parameter, optimized)
                self.assertIsNone(parameter.grad)

            second = train_step(continuous, optimizer, loss_function, inputs, targets)
            self.assertEqual(second, train_step(restored, resumed, loss_function, inputs, targets))
            after = optimizer_record(resumed, optimizer_path)
            self.assertEqual(after, optimizer_record(optimizer, Path(folder) / "continuous.json"))
            self.assertNotEqual(before, after)
            self.assertTrue(all(state["step"] == 2 for state in after["state"]))

        self.assertTrue(all(isfinite(value) for record in (first, second) for value in record))
        self.assertGreater(first[1], 0.25)
        self.assertNotEqual(initial, [parameter.values for parameter in restored.parameters()])
        self.assertEqual([parameter.values for parameter in continuous.parameters()],
                         [parameter.values for parameter in restored.parameters()])
        self.assertEqual(continuous(inputs, train=False).tolist(), restored(inputs, train=False).tolist())

    def test_optimizer_load_keeps_model_values_gradients_and_parameter_references(self):
        source = Tensor([2, -1], requires_grad=True)
        optimizer = AdamW([source], learning_rate=0.02, betas=(0.7, 0.9), eps=1e-6, weight_decay=0.03)
        source.square().sum().backward()
        optimizer.step()
        target = Tensor([123, 456], requires_grad=True)
        gradient = target.grad = Tensor([7, 8])
        restored = AdamW([target])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "optimizer.json"
            expected = optimizer_record(optimizer, path)
            restored.load(path)
            self.assertEqual(expected, optimizer_record(restored, Path(folder) / "restored.json"))
        self.assertIs(restored.parameters[0], target)
        self.assertEqual(target.values, [123, 456])
        self.assertIs(target.grad, gradient)
        self.assertEqual(gradient.values, [7, 8])
        self.assertTrue(target.requires_grad)

    def test_clip_between_backward_and_step_changes_only_the_gradient(self):
        parameter = Tensor([3, 4], requires_grad=True)
        optimizer = nn.SGD([parameter], learning_rate=0.5)
        parameter.square().sum().backward()
        self.assertEqual(parameter.grad.values, [6, 8])
        self.assertEqual(clip_grad_norm([parameter, parameter], max_norm=2), 10)
        self.assertEqual(parameter.values, [3, 4])
        self.assertAlmostEqual(parameter.grad.values[0], 1.2)
        self.assertAlmostEqual(parameter.grad.values[1], 1.6)
        self.assertAlmostEqual(sqrt(sum(value * value for value in parameter.grad.values)), 2)
        optimizer.step()
        self.assertAlmostEqual(parameter.values[0], 2.4)
        self.assertAlmostEqual(parameter.values[1], 3.2)

    def test_small_language_model_two_steps_with_paired_save_and_load(self):
        model = lm.Model(lm.Tokenizer("abc"), context_length=3, features=4, heads=2,
                         hidden_features=6, depth=1, seed=7)
        optimizer = AdamW(model, learning_rate=0.01)
        loss_function = nn.CrossEntropy()
        inputs, targets = [[0, 1, 2]], [[1, 2, 0]]
        initial = [parameter.values[:] for parameter in model.parameters()]
        first = train_step(model, optimizer, loss_function, inputs, targets)
        prediction = model(inputs, train=False).tolist()
        with TemporaryDirectory() as folder:
            model_path, optimizer_path = Path(folder) / "language.json", Path(folder) / "optimizer.json"
            model.save(model_path)
            optimizer.save(optimizer_path)
            model = lm.Model().load(model_path)
            optimizer = AdamW(model).load(optimizer_path)
        self.assertEqual(prediction, model(inputs, train=False).tolist())
        second = train_step(model, optimizer, loss_function, inputs, targets)
        final_loss = loss_function(model(inputs, train=False), targets).item()
        self.assertTrue(all(isfinite(value) for record in (first, second) for value in record))
        self.assertTrue(isfinite(final_loss))
        self.assertEqual(model(inputs, train=False).shape, (1, 3, 3))
        self.assertNotEqual(initial, [parameter.values for parameter in model.parameters()])
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(all(isfinite(value) for value in parameter.grad.values))


if __name__ == "__main__":
    unittest.main()
