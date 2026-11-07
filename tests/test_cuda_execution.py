import gc
import math
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from qutai import Tensor, is_available, nn
from qutai.backends import cpu, cuda
from qutai.lm.precision import autocast


def rounded_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class ParameterGroupTests(unittest.TestCase):
    def test_groups_decay_and_round_trip(self):
        a, b = Tensor([1.0], requires_grad=True), Tensor([1.0], requires_grad=True)
        opt = nn.AdamW(
            [{"params": [a], "weight_decay": 0.1}, {"params": [b], "weight_decay": 0}],
            learning_rate=0.2,
        )
        a.grad, b.grad = Tensor([0.0]), Tensor([0.0])
        opt.step()
        self.assertAlmostEqual(a.item(), 0.98)
        self.assertEqual(b.item(), 1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "optimizer.json"
            opt.save(path)
            restored = nn.AdamW([a, b]).load(path)
            self.assertEqual(
                [g["weight_decay"] for g in restored.param_groups], [0.1, 0]
            )
            self.assertIs(restored.param_groups[0]["params"][0], a)
        with self.assertRaises(ValueError):
            nn.AdamW([{"params": [a]}, {"params": [a]}])


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class CUDAExecutionTests(unittest.TestCase):
    def test_dynamic_shape_cache_has_a_budget(self):
        runtime = cuda.get_runtime()
        limit = runtime.cache_limit_bytes
        cuda.empty_cache()
        try:
            runtime.cache_limit_bytes = 128 << 10
            for size in range(17000, 27000, 191):
                value = cuda.full(size, 2, dtype="float32")
                self.assertEqual(value.tolist()[0], 2)
                del value
            stats = cuda.memory_stats()
            self.assertLessEqual(
                stats["reserved_bytes"] - stats["allocated_bytes"],
                runtime.cache_limit_bytes,
            )
        finally:
            runtime.cache_limit_bytes = limit
            cuda.empty_cache()

    def test_async_pool_reuses_after_last_python_reference(self):
        gc.collect()
        start = cuda.memory_stats()["allocated_bytes"]
        x = cuda.full(8192, 1, dtype="float32")
        for _ in range(100):
            x = cuda.add(x, cuda.full(x.size, 1, dtype=x.dtype))
        self.assertEqual(x.tolist(), [101.0] * x.size)
        reserved = cuda.memory_stats()["reserved_bytes"]
        del x
        gc.collect()
        self.assertEqual(cuda.memory_stats()["allocated_bytes"], start)
        for _ in range(100):
            value = cuda.full(8192, 2, dtype="float32")
            del value
        self.assertEqual(cuda.memory_stats()["reserved_bytes"], reserved)

    def test_gemm_transpose_broadcast_and_non_tile_shapes(self):
        for dtype in ("float32", "float64"):
            for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
                m, k, n = 5, 35, 17
                a = [math.sin(i) for i in range(m * k)]
                b = [math.cos(i) for i in range(k * n)]
                shape_a, shape_b = (
                    ((k, m) if ta else (m, k)),
                    ((n, k) if tb else (k, n)),
                )
                ca, cb = cpu.storage(a, dtype=dtype), cpu.storage(b, dtype=dtype)
                if ta:
                    ca = cpu.transpose(ca, shape_a)
                if tb:
                    cb = cpu.transpose(cb, shape_b)
                expected = cpu.matmul(ca, cb, (m, k), (k, n)).tolist()
                result = cuda.gemm(
                    cuda.storage(a, dtype=dtype),
                    cuda.storage(b, dtype=dtype),
                    m,
                    k,
                    n,
                    transpose_left=ta,
                    transpose_right=tb,
                ).tolist()
                self.assertLess(
                    max(abs(x - y) for x, y in zip(result, expected)),
                    4e-6 if dtype == "float32" else 1e-12,
                )

    def test_bf16_tensor_core_uses_fp32_accumulation(self):
        m, k, n = 17, 35, 19
        a, b = (
            [math.sin(i * 0.17) for i in range(m * k)],
            [math.cos(i * 0.13) for i in range(k * n)],
        )
        expected = cpu.matmul(
            cpu.storage(map(rounded_bf16, a), dtype="float32"),
            cpu.storage(map(rounded_bf16, b), dtype="float32"),
            (m, k),
            (k, n),
        ).tolist()
        before = cuda.get_runtime().tensor_core_launches
        with autocast():
            result = cuda.gemm(
                cuda.storage(a, dtype="float32"),
                cuda.storage(b, dtype="float32"),
                m,
                k,
                n,
            )
        self.assertEqual(result.dtype, "float32")
        self.assertGreater(cuda.get_runtime().tensor_core_launches, before)
        self.assertLess(
            max(abs(x - y) for x, y in zip(result.tolist(), expected)), 3e-6
        )

    def test_device_initialization_and_in_place_adam(self):
        a = cuda.normal(32768, std=0.02, seed=17)
        b = cuda.normal(32768, std=0.02, seed=17)
        self.assertEqual(a.tolist(), b.tolist())
        values = a.tolist()
        for seed in (18, (17 << 32) + 17):
            other = cuda.normal(len(values), std=0.02, seed=seed).tolist()
            mx, my = sum(values) / len(values), sum(other) / len(other)
            correlation = sum(
                (x - mx) * (y - my) for x, y in zip(values, other)
            ) / math.sqrt(
                sum((x - mx) ** 2 for x in values) * sum((y - my) ** 2 for y in other)
            )
            self.assertLess(abs(correlation), 0.04)
        self.assertLess(abs(sum(values) / len(values)), 0.001)
        self.assertAlmostEqual(
            math.sqrt(sum(x * x for x in values) / len(values)), 0.02, delta=0.001
        )
        model = nn.Model([nn.Linear(2, 1, seed=1)]).to("cuda", dtype="float32")
        opt = nn.AdamW(model)
        pointers = [p.storage.ptr for p in model.parameters()]
        for _ in range(2):
            opt.zero_grad()
            with autocast():
                loss = nn.MSE()(model([[1, 2]]), [[0]])
            loss.backward(retain_graph=False)
            opt.step()
        self.assertEqual(pointers, [p.storage.ptr for p in model.parameters()])
        self.assertTrue(all(x.dtype == "float32" for x in opt.averages.values()))
