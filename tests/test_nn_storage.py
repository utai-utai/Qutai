"""训练状态、缓存和模型文件统一经过数值存储接口。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qutai import Tensor, nn
from qutai.backends.cpu import CPUStorage
from qutai.backends.storage import Storage
from qutai.nn.cache import KVCache


class NeuralStorageTests(unittest.TestCase):
    def make_model(self):
        return nn.Model([nn.BatchNorm2d(1), nn.Flatten(), nn.Linear(2, 1, seed=7)])

    def test_training_keeps_parameters_gradients_batch_stats_and_optimizer_moments_in_storage(self):
        model = self.make_model()
        optimizer = nn.AdamW(model, learning_rate=0.01)
        normalizer = model.layers[0]
        mean, variance = normalizer.running_mean, normalizer.running_var
        inputs, targets = [[[[1, 2]]], [[[3, 5]]]], [[0], [1]]
        for _ in range(2):
            optimizer.zero_grad()
            nn.MSE()(model(inputs), targets).backward()
            nn.clip_grad_norm(model)
            optimizer.step()
        self.assertIs(normalizer.running_mean, mean)
        self.assertIs(normalizer.running_var, variance)
        for value in (mean, variance, *optimizer.averages.values(), *optimizer.square_averages.values()):
            self.assertIsInstance(value, Storage)
            self.assertEqual((value.device, value.dtype), ("cpu", "float64"))
        for parameter in model.parameters():
            self.assertIsInstance(parameter.storage, Storage)
            self.assertIsInstance(parameter.grad.storage, Storage)
            self.assertEqual(optimizer.steps[parameter], 2)

    def test_forward_backward_optimizer_and_kv_cache_do_not_export_numeric_lists(self):
        model, cache = self.make_model(), KVCache()
        optimizer = nn.SGD(model, momentum=0.9)
        first, second = Tensor(list(range(8)), shape=(1, 2, 2, 2)), Tensor(list(range(4)), shape=(1, 2, 1, 2))
        with patch.object(CPUStorage, "tolist", side_effect=AssertionError("计算过程不应导出列表")):
            loss = nn.MSE()(model([[[[1, 2]]], [[[3, 5]]]]), [[0], [1]])
            loss.backward()
            nn.clip_grad_norm(model)
            optimizer.step()
            cache.update(first, first)
            keys, values = cache.update(second, second)
        self.assertEqual(keys.shape, (1, 2, 3, 2))
        self.assertEqual(values.shape, keys.shape)
        self.assertTrue(all(isinstance(value, Storage) for value in optimizer.velocities.values()))
        self.assertEqual(keys.tolist(), [[[[0, 1], [2, 3], [0, 1]], [[4, 5], [6, 7], [2, 3]]]])

    def test_cross_entropy_backward_uses_storage_without_export(self):
        scores = Tensor([[1, 2, 3], [0.5, -0.5, 1]], requires_grad=True)
        with patch.object(CPUStorage, "tolist", side_effect=AssertionError("交叉熵不应导出列表")):
            loss = nn.CrossEntropy()(scores, [2, 0])
            loss.backward()
        self.assertIsInstance(scores.grad.storage, Storage)
        for row in scores.grad.tolist():
            self.assertAlmostEqual(sum(row), 0)

    def test_model_and_optimizer_roundtrip_preserves_shared_state_freezing_and_schema(self):
        normalizer, output = nn.BatchNorm2d(1), nn.Linear(2, 1, seed=7)
        output.bias.requires_grad = False
        model = nn.Model([normalizer, normalizer, nn.Flatten(), output])
        optimizer = nn.AdamW(model)
        inputs, targets = [[[[1, 2]]], [[[3, 5]]]], [[0], [1]]
        nn.MSE()(model(inputs), targets).backward()
        optimizer.step()
        before = model(inputs, train=False).tolist()
        with TemporaryDirectory() as folder:
            model_path, optimizer_path = Path(folder) / "model.json", Path(folder) / "optimizer.json"
            model.save(model_path)
            optimizer.save(optimizer_path)
            record = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(set(record), {"model", "parameters"})
            self.assertTrue(all(isinstance(parameter["values"], list) for parameter in record["parameters"]))
            restored = nn.Model().load(model_path)
            resumed = nn.AdamW(restored).load(optimizer_path)
        self.assertIs(restored.layers[0], restored.layers[1])
        self.assertIsInstance(restored.layers[0].running_mean, Storage)
        self.assertFalse(restored.layers[-1].bias.requires_grad)
        self.assertEqual(restored(inputs, train=False).tolist(), before)
        for left, right in zip(optimizer.parameters, resumed.parameters, strict=True):
            if left in optimizer.averages:
                self.assertEqual(optimizer.averages[left], resumed.averages[right])
                self.assertEqual(optimizer.square_averages[left], resumed.square_averages[right])
                self.assertIsInstance(resumed.averages[right], Storage)


if __name__ == "__main__":
    unittest.main()
