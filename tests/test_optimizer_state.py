import json
import tempfile
import unittest
from pathlib import Path

from qutai import Tensor
from qutai.nn.optimizers import AdamW, SGD


class OptimizerStateTests(unittest.TestCase):
    def test_adamw_resume_restores_settings_moments_and_per_parameter_steps(self):
        first, second, unused = Tensor([1, 2]), Tensor(3), Tensor(4)
        optimizer = AdamW([first, second, first, unused], learning_rate=0.02, betas=(0.6, 0.8), eps=0.01, weight_decay=0.2)
        first.grad, second.grad = Tensor([2, -1]), Tensor(0)
        optimizer.step()
        second.zero_grad()
        first.grad = Tensor([1, 3])
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            optimizer.save(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["type"], "AdamW")
            self.assertNotIn("version", data)
            self.assertNotIn("parameters", data)
            self.assertEqual(data["state"][0]["step"], 2)
            self.assertEqual(data["state"][1]["step"], 1)
            self.assertIsNone(data["state"][2])
            restored_first, restored_second, restored_unused = Tensor(first.tolist()), Tensor(second.item()), Tensor(unused.item())
            restored = AdamW([restored_first, restored_second, restored_unused])
            sentinel = Tensor([7, 8])
            restored_first.grad = sentinel
            self.assertIs(restored.load(path), restored)
            self.assertEqual(restored.learning_rate, 0.02)
            self.assertEqual(restored.betas, (0.6, 0.8))
            self.assertEqual(restored.eps, 0.01)
            self.assertEqual(restored.weight_decay, 0.2)
            self.assertIs(restored.parameters[0], restored_first)
            self.assertIs(restored_first.grad, sentinel)
            self.assertEqual(restored_first.tolist(), first.tolist())
            restored.save(Path(directory) / "again.json")
            self.assertEqual(json.loads((Path(directory) / "again.json").read_text()), data)

            first.grad = Tensor([-3, 2])
            second.grad = Tensor(5)
            restored_first.grad = Tensor([-3, 2])
            restored_second.grad = Tensor(5)
            optimizer.step()
            restored.step()
            self.assertEqual(restored_first.tolist(), first.tolist())
            self.assertEqual(restored_second.item(), second.item())
            self.assertEqual(restored_unused.item(), unused.item())
            self.assertEqual(restored.steps[restored_first], 3)
            self.assertEqual(restored.steps[restored_second], 2)

    def test_sgd_resume_restores_momentum_and_all_settings(self):
        parameter = Tensor([2, -1])
        optimizer = SGD([parameter], learning_rate=0.15, momentum=0.8, weight_decay=0.2)
        parameter.grad = Tensor([1, -2])
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            optimizer.save(path)
            restored_parameter = Tensor(parameter.tolist())
            restored = SGD([restored_parameter]).load(path)
            self.assertEqual(restored.learning_rate, 0.15)
            self.assertEqual(restored.momentum, 0.8)
            self.assertEqual(restored.weight_decay, 0.2)
            self.assertEqual(restored.velocities[restored_parameter], optimizer.velocities[parameter])
            parameter.grad = Tensor([3, 4])
            restored_parameter.grad = Tensor([3, 4])
            optimizer.step()
            restored.step()
            self.assertEqual(restored_parameter.tolist(), parameter.tolist())
            self.assertEqual(restored.velocities[restored_parameter], optimizer.velocities[parameter])

    def test_loading_uses_current_parameter_references_without_model_values(self):
        original = Tensor(2)
        optimizer = SGD([original], momentum=0.9)
        original.grad = Tensor(3)
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            optimizer.save(path)
            current = Tensor(100)
            current.grad = Tensor(5)
            gradient = current.grad
            restored = SGD([current, current]).load(path)
            self.assertEqual(current.item(), 100)
            self.assertIs(current.grad, gradient)
            self.assertEqual(restored.parameters, (current,))
            self.assertEqual(restored.velocities[current], [3])

    def test_empty_and_unstarted_optimizers_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            for kind in (SGD, AdamW):
                for parameters in ([], [Tensor([1, 2])]):
                    with self.subTest(kind=kind, parameters=parameters):
                        optimizer = kind(parameters)
                        optimizer.save(path)
                        saved = json.loads(path.read_text())
                        self.assertEqual(saved["state"], [None] * len(parameters))
                        restored = kind(parameters).load(path)
                        restored.save(path)
                        self.assertEqual(json.loads(path.read_text()), saved)

    def test_wrong_type_count_shape_and_state_rejected_without_partial_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            source = Tensor([1, 2])
            source.grad = Tensor([2, 3])
            original = AdamW([source], learning_rate=0.04)
            original.step()
            original.save(path)
            data = json.loads(path.read_text())
            wrong_type = SGD([Tensor([1, 2])])
            with self.assertRaisesRegex(ValueError, "类型"):
                wrong_type.load(path)
            for parameters in ([], [Tensor([[1, 2]])]):
                with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, "数量或形状"):
                    AdamW(parameters).load(path)
            parameter = Tensor([9, 8])
            parameter.grad = Tensor([4, 5])
            optimizer = AdamW([parameter], learning_rate=0.2)
            optimizer.step()
            before_path = Path(directory) / "before.json"
            optimizer.save(before_path)
            before = before_path.read_text()
            for update in ({"state": []}, {"state": [{"step": 1, "average": [], "square_average": [1, 2]}]},
                           {"state": [{"step": 0, "average": [1, 2], "square_average": [1, 2]}]},
                           {"state": [{"step": 1, "average": [1, 2], "square_average": [-1, 2]}]}):
                path.write_text(json.dumps({**data, **update}))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    optimizer.load(path)
                optimizer.save(before_path)
                self.assertEqual(before_path.read_text(), before)
                self.assertIs(optimizer.parameters[0], parameter)

    def test_sgd_invalid_velocity_length_is_rejected(self):
        parameter = Tensor([1, 2])
        optimizer = SGD([parameter], momentum=0.8)
        parameter.grad = Tensor([2, 3])
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            optimizer.save(path)
            data = json.loads(path.read_text())
            data["state"] = [[1]]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "长度"):
                optimizer.load(path)
            self.assertEqual(optimizer.velocities[parameter], [2, 3])


if __name__ == "__main__":
    unittest.main()
