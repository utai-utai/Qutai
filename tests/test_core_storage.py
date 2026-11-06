"""Tensor 的存储所有权、计算结果与反向传播数据边界。"""

import unittest
from unittest.mock import patch

from qutai import Tensor, stack
from qutai.backends import cpu
from qutai.backends.storage import Storage
from qutai.core import operations


class TensorStorageTests(unittest.TestCase):
    def test_lists_are_copied_but_internal_storage_results_are_adopted(self):
        values = [[1, 2], [3, 4]]
        tensor = Tensor(values)
        values[0][0] = 99
        self.assertEqual(tensor.tolist(), [[1, 2], [3, 4]])
        self.assertIsInstance(tensor.storage, Storage)
        self.assertEqual((tensor.device, tensor.dtype, tensor.size), ("cpu", "float64", 4))
        self.assertNotIn("device", vars(tensor))
        self.assertNotIn("dtype", vars(tensor))
        raw = cpu.storage([1, 2])
        adopted = Tensor(raw, shape=(2,))
        self.assertIs(adopted.storage, raw)
        copied = Tensor(raw.copy(), shape=(2,))
        raw[0] = 7
        self.assertEqual(copied.tolist(), [1, 2])

    def test_reshape_and_indexing_keep_independent_storage(self):
        tensor = Tensor([[1, 2], [3, 4]], requires_grad=True)
        reshaped, selected = tensor.reshape((4,)), tensor[:, 0]
        self.assertIsNot(reshaped.storage, tensor.storage)
        self.assertIsNot(selected.storage, tensor.storage)
        reshaped.values[0] = 99
        selected.values[1] = 88
        self.assertEqual(tensor.tolist(), [[1, 2], [3, 4]])
        reshaped.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [[1, 1], [1, 1]])

    def test_shared_seed_does_not_alias_leaf_gradients_or_input_seed(self):
        left, right = Tensor([1, 2], requires_grad=True), Tensor([3, 4], requires_grad=True)
        seed = Tensor([2, 3])
        (left + right).backward(seed)
        self.assertIsNot(left.grad.storage, right.grad.storage)
        self.assertIsNot(left.grad.storage, seed.storage)
        left.grad.values[0] = 99
        self.assertEqual(right.grad.tolist(), [2, 3])
        self.assertEqual(seed.tolist(), [2, 3])

    def test_composite_forward_and_backward_keep_vectors_in_storage(self):
        tensor = Tensor([[0.2, -0.4], [0.7, 0.1]], requires_grad=True)
        bias = Tensor([0.3, 0.6], requires_grad=True)
        with patch.object(cpu.CPUStorage, "tolist", side_effect=AssertionError("计算过程不能导出列表")):
            result = stack([tensor + bias, tensor * bias], axis=1)
            loss = ((result.square() + 1).log().exp() / 2).mean()
            loss.backward()
        self.assertIsInstance(loss.storage, Storage)
        self.assertIsInstance(tensor.grad.storage, Storage)
        self.assertIsInstance(bias.grad.storage, Storage)
        self.assertEqual(tensor.grad.shape, tensor.shape)

    def test_cpu_debug_alias_is_live_and_exports_are_independent(self):
        tensor = Tensor([1, 2])
        self.assertIs(tensor.values, tensor.storage)
        tensor.values[0] = 3
        self.assertEqual(tensor.tolist(), [3, 2])
        output = tensor.tolist()
        output[0] = 99
        self.assertEqual(tensor.tolist(), [3, 2])
        tensor.values = [4, 5]
        self.assertEqual(tensor.tolist(), [4, 5])
        self.assertEqual(tensor.dtype, "float64")

    def test_public_device_dtype_and_shape_contracts_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "设备"):
            Tensor([1], device="cuda")
        with self.assertRaisesRegex(ValueError, "float64"):
            Tensor([1], dtype="float16")
        with self.assertRaises(ValueError):
            Tensor([1, 2], shape=(3,))
        with self.assertRaises(ValueError):
            Tensor(cpu.storage([1j], dtype="complex128"), shape=(1,))
        right = Tensor([1])
        right.storage.dtype = "complex128"
        with self.assertRaisesRegex(ValueError, "精度"):
            operations.add(Tensor([1]), right)


if __name__ == "__main__":
    unittest.main()
