"""CUDA 空形状、重复索引、稳定损失、精度转换与显存生命周期。"""

import gc
from math import hypot, isfinite
import unittest
from unittest.mock import patch

from qutai import Tensor, is_available, nn
from qutai.backends import cpu, cuda


@unittest.skipUnless(is_available("cuda"), "需要实际 CUDA 设备与 NVRTC")
class CUDAEdgeTests(unittest.TestCase):
    def assertValuesClose(self, actual, expected, tolerance=2e-6):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True): self.assertLessEqual(abs(a-b), tolerance * (1 + abs(b)))

    def test_empty_broadcast_transpose_and_zero_inner_matmul_gradients(self):
        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取列表")), \
             patch.object(cpu, "matmul", side_effect=AssertionError("不能回退 CPU")):
            x = Tensor([], shape=(0, 2, 3), device="cuda", dtype="float32", requires_grad=True)
            bias = Tensor([1, 2, 3], shape=(1, 1, 3), device="cuda", dtype="float32", requires_grad=True)
            result = (x + bias).transpose((2, 0, 1))
            self.assertEqual(result.shape, (3, 0, 2))
            result.sum().backward()
            a = Tensor([], shape=(2, 0), device="cuda", dtype="float32", requires_grad=True)
            b = Tensor([], shape=(0, 3), device="cuda", dtype="float32", requires_grad=True)
            product = a @ b
            self.assertEqual(product.shape, (2, 3))
            product.sum().backward()
            batched = Tensor([], shape=(0, 2, 3), device="cuda", dtype="float32", requires_grad=True)
            weight = Tensor(list(range(12)), shape=(1, 3, 4), device="cuda", dtype="float32", requires_grad=True)
            empty_product = batched @ weight
            self.assertEqual(empty_product.shape, (0, 2, 4))
            empty_product.sum().backward()
        self.assertEqual(x.grad.tolist(), [])
        self.assertEqual(bias.grad.storage.tolist(), [0, 0, 0])
        self.assertEqual(product.storage.tolist(), [0] * 6)
        self.assertEqual((a.grad.size, b.grad.size), (0, 0))
        self.assertEqual(weight.grad.storage.tolist(), [0] * 12)

    def test_vector_and_broadcast_batch_matmul_gradients(self):
        for dtype in ("float32", "float64"):
            for left_shape, right_shape in (((3,), (3,)), ((3,), (2, 3, 2)), ((2, 1, 2, 3), (1, 4, 3, 2))):
                size_a, size_b = 1, 1
                for d in left_shape: size_a *= d
                for d in right_shape: size_b *= d
                values_a, values_b = [(i-3)/7 for i in range(size_a)], [(i+1)/11 for i in range(size_b)]
                reference_a = Tensor(values_a, shape=left_shape, dtype=dtype, requires_grad=True)
                reference_b = Tensor(values_b, shape=right_shape, dtype=dtype, requires_grad=True)
                reference = reference_a @ reference_b
                reference.square().sum().backward()
                a = Tensor(values_a, shape=left_shape, dtype=dtype, device="cuda", requires_grad=True)
                b = Tensor(values_b, shape=right_shape, dtype=dtype, device="cuda", requires_grad=True)
                with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取列表")), \
                     patch.object(cpu, "matmul", side_effect=AssertionError("不能回退 CPU")):
                    actual = a @ b
                    actual.square().sum().backward()
                self.assertEqual(actual.shape, reference.shape)
                self.assertValuesClose(actual.storage.tolist(), reference.storage.tolist())
                self.assertValuesClose(a.grad.storage.tolist(), reference_a.grad.storage.tolist())
                self.assertValuesClose(b.grad.storage.tolist(), reference_b.grad.storage.tolist())

    def test_repeated_embedding_indices_accumulate_gradients(self):
        for dtype in ("float32", "float64"):
            reference = nn.Embedding(3, 2, seed=7).to(dtype=dtype)
            model = nn.Embedding(3, 2, seed=7).to("cuda", dtype=dtype)
            expected = reference([[1, 1], [2, 1]])
            expected.sum().backward()
            with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取列表")), \
                 patch.object(cpu, "gather", side_effect=AssertionError("不能回退 CPU")), \
                 patch.object(cpu, "scatter", side_effect=AssertionError("不能回退 CPU")):
                actual = model([[1, 1], [2, 1]])
                actual.sum().backward()
            self.assertValuesClose(actual.storage.tolist(), expected.storage.tolist())
            self.assertEqual(model.weight.grad.tolist(), [[0, 0], [3, 3], [1, 1]])

    def test_large_logits_cross_entropy_is_stable(self):
        values = [[10000, 10001, -10000], [10000, -10000, 10000]]
        for dtype in ("float32", "float64"):
            reference_x = Tensor(values, dtype=dtype, requires_grad=True)
            reference = nn.CrossEntropy()(reference_x, [0, 2])
            reference.backward()
            x = Tensor(values, dtype=dtype, device="cuda", requires_grad=True)
            with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取列表")), \
                 patch.object(cpu, "cross_entropy", side_effect=AssertionError("不能回退 CPU")):
                actual = nn.CrossEntropy()(x, [0, 2])
                actual.backward()
            self.assertTrue(isfinite(actual.item()))
            self.assertValuesClose(actual.storage.tolist(), reference.storage.tolist())
            self.assertValuesClose(x.grad.storage.tolist(), reference_x.grad.storage.tolist())

    def test_device_cast_and_copy_preserve_complex_components_and_independence(self):
        values = [1.25-0.5j, -2+4j, 0.125+0.25j]
        original = cuda.storage(values, dtype="complex64")
        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("设备转换不能读取列表")):
            converted = cuda.from_storage(original, dtype="complex128")
            clone = converted.copy()
            cuda.copy_into(clone, cuda.full(3, 9+8j, dtype="complex128"))
        self.assertEqual((original.nbytes, converted.nbytes), (24, 48))
        self.assertEqual(converted.tolist(), values)
        self.assertEqual(original.tolist(), values)
        self.assertEqual(clone.tolist(), [9+8j] * 3)

    def test_storage_layout_transpose_and_broadcast_preserve_whole_elements(self):
        for dtype, values in (("complex64", [1+2j, 3-4j]), ("complex128", [1+2j, 3-4j]), ("int64", [2**60+1, -2**60-3])):
            data = cuda.storage(values, dtype=dtype)
            with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("布局转换不能读取列表")):
                expanded = cuda.broadcast(data, (2, 1), (2, 3))
                transposed = cuda.transpose(expanded, (2, 3))
            self.assertEqual(expanded.tolist(), [values[0]] * 3 + [values[1]] * 3)
            self.assertEqual(transposed.tolist(), values * 3)

    def test_global_norm_extremes_remain_finite_without_reading_whole_vectors(self):
        for dtype, values in (("float32", [1e30, -1e30, 1e-30]), ("float64", [1e300, -1e300, 1e-300])):
            data = cuda.storage(values, dtype=dtype)
            expected = hypot(*data.tolist())
            read = cuda.CUDAStorage.tolist
            def scalar_only(storage):
                self.assertEqual(storage.size, 1, "范数只能读取归约后的标量")
                return read(storage)
            with patch.object(cuda.CUDAStorage, "tolist", scalar_only): actual = cuda.global_norm([data])
            self.assertTrue(isfinite(actual))
            self.assertAlmostEqual(actual / expected, 1, places=6)

    def test_temporary_storage_is_reclaimed(self):
        runtime = cuda.get_runtime()
        gc.collect()
        before = len(runtime._allocations)
        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取列表")):
            for _ in range(20):
                x = Tensor([1, 2, 3], device="cuda", dtype="float32")
                y = x.to(dtype="float64", copy=True)
                z = (y + 2).square().sum()
                del x, y, z
        gc.collect()
        self.assertEqual(len(runtime._allocations), before)


if __name__ == "__main__": unittest.main()
