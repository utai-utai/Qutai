import math
import struct
import unittest
from unittest.mock import patch

from qutai import Tensor, is_available
from qutai.backends import cpu, cuda


def rounded(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7FFFFFFF > 0x7F800000:
        bits = (bits >> 16) | 0x40
    else:
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class BFloat16StorageTests(unittest.TestCase):
    def test_host_and_gpu_cast_rounding_use_two_bytes(self):
        values = [0, -0.0, 1.00390625, 1.01171875, -2.71828, 1e-30, 1e30]
        expected = list(map(rounded, values))
        host = cuda.storage(values, dtype="bfloat16")
        device = cuda.from_storage(
            cuda.storage(values, dtype="float32"), dtype="bfloat16"
        )
        self.assertEqual(host.nbytes, len(values) * 2)
        self.assertEqual(host.tolist(), expected)
        self.assertEqual(device.tolist(), expected)
        self.assertEqual(cuda.from_storage(device, dtype="float32").tolist(), expected)
        self.assertEqual(device.copy().tolist(), expected)
        special = cuda.storage(
            [math.inf, -math.inf, math.nan], dtype="bfloat16"
        ).tolist()
        self.assertEqual(special[:2], [math.inf, -math.inf])
        self.assertTrue(math.isnan(special[2]))

    def test_same_device_cast_is_queued_without_explicit_sync(self):
        x = Tensor([1, 2], device="cuda", dtype="float32", requires_grad=True)
        with patch.object(cuda, "synchronize", side_effect=AssertionError("不应同步")):
            y = x.to(dtype="bfloat16")
            result = y.to(dtype="float32")
        self.assertEqual(result.tolist(), [1, 2])
        result.sum().backward(retain_graph=False)
        self.assertEqual(x.grad.dtype, "float32")
        self.assertEqual(x.grad.tolist(), [1, 1])

    def test_basic_math_broadcast_reduce_and_gradients(self):
        x = Tensor(
            [[1, 2], [3, 4]], device="cuda", dtype="bfloat16", requires_grad=True
        )
        loss = ((x * x) + [1, 2]).mean()
        self.assertEqual(loss.dtype, "bfloat16")
        self.assertEqual(loss.item(), 9)
        loss.backward(retain_graph=False)
        self.assertEqual(x.grad.tolist(), [[0.5, 1], [1.5, 2]])
        self.assertEqual(x.transpose().tolist(), [[1, 3], [2, 4]])
        self.assertEqual(x.argmax(axis=1), [1, 1])

    def test_in_place_helpers_do_not_allocate_new_storage(self):
        for dtype in ("bfloat16", "float32", "float64"):
            x = cuda.storage([1, 2, 3], dtype=dtype)
            pointer = x.ptr
            cuda.add_into(x, cuda.storage([2, 3, 4], dtype=dtype))
            cuda.scale_into(x, 0.5)
            self.assertEqual(x.ptr, pointer)
            self.assertEqual(x.tolist(), [1.5, 2.5, 3.5])

    def test_matmul_backward_composes_with_bfloat16_nonleaf(self):
        x = Tensor(
            [[1, 2], [3, 4]], device="cuda", dtype="bfloat16", requires_grad=True
        )
        weight = Tensor([[1], [2]], device="cuda", dtype="bfloat16", requires_grad=True)
        loss = ((x * 2) @ weight).square().mean()
        self.assertEqual(loss.item(), 292)
        loss.backward(retain_graph=False)
        self.assertEqual(x.grad.dtype, "bfloat16")
        self.assertEqual(x.grad.tolist(), [[20, 40], [44, 88]])
        self.assertEqual(weight.grad.tolist(), [[152], [216]])

    def test_cpu_rejection_and_explicit_cpu_float_conversion(self):
        with self.assertRaises(ValueError):
            Tensor([1], dtype="bfloat16")
        x = Tensor([1.25], device="cuda", dtype="bfloat16")
        self.assertEqual(x.to("cpu", dtype="float32").tolist(), [1.25])
        self.assertIsInstance(x.to("cpu", dtype="float32").storage, cpu.CPUStorage)
        with self.assertRaises(ValueError):
            cuda.real_inner(x.storage, x.storage)
