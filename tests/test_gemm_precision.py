import math
import struct
import unittest

from qutai import is_available
from qutai.backends import cuda
from qutai.lm.precision import autocast


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def reference(a, b, m, k, n, ta=False, tb=False):
    return [
        sum(
            a[p * m + r if ta else r * k + p] * b[c * k + p if tb else p * n + c]
            for p in range(k)
        )
        for r in range(m)
        for c in range(n)
    ]


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class GEMMPrecisionTests(unittest.TestCase):
    def assertValues(self, actual, expected, tolerance=4e-6):
        self.assertEqual(len(actual), len(expected))
        self.assertLessEqual(
            max((abs(a - b) for a, b in zip(actual, expected)), default=0), tolerance
        )

    def test_mixed_inputs_all_transposes_outside_autocast(self):
        m, k, n = 5, 35, 17
        a = [math.sin(i * 0.17) for i in range(m * k)]
        b = [math.cos(i * 0.13) for i in range(k * n)]
        expected_a, expected_b = list(map(bf16, a)), list(map(bf16, b))
        for da, db in (
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
            ("bfloat16", "bfloat16"),
        ):
            left, right = cuda.storage(a, dtype=da), cuda.storage(b, dtype=db)
            self.assertEqual(left.nbytes, len(a) * (2 if da == "bfloat16" else 4))
            for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
                before = cuda.get_runtime().tensor_core_launches
                result = cuda.gemm(
                    left, right, m, k, n, transpose_left=ta, transpose_right=tb
                )
                self.assertEqual(result.dtype, "float32")
                self.assertEqual(cuda.get_runtime().tensor_core_launches, before + 1)
                self.assertValues(
                    result.tolist(), reference(expected_a, expected_b, m, k, n, ta, tb)
                )

    def test_bf16_output_and_in_place_accumulation(self):
        m, k, n = 17, 35, 19
        a = [math.sin(i * 0.17) for i in range(m * k)]
        b = [math.cos(i * 0.13) for i in range(k * n)]
        left, right = (
            cuda.storage(a, dtype="bfloat16"),
            cuda.storage(b, dtype="float32"),
        )
        expected = reference(list(map(bf16, a)), list(map(bf16, b)), m, k, n)
        target = cuda.full(m * n, 0.25, dtype="float32")
        pointer = target.ptr
        result = cuda.gemm(left, right, m, k, n, out=target)
        self.assertIs(result, target)
        self.assertEqual(target.ptr, pointer)
        self.assertValues(target.tolist(), expected)
        cuda.gemm(left, right, m, k, n, out=target, accumulate=True)
        self.assertValues(target.tolist(), [2 * x for x in expected], 8e-6)
        rounded = cuda.gemm(left, right, m, k, n, out_dtype="bfloat16")
        self.assertEqual(rounded.nbytes, m * n * 2)
        self.assertValues(rounded.tolist(), list(map(bf16, expected)), 0)
        old = rounded.tolist()
        cuda.gemm(
            left, right, m, k, n, out_dtype="bfloat16", out=rounded, accumulate=True
        )
        self.assertValues(
            rounded.tolist(), [bf16(x + y) for x, y in zip(old, expected)], 0
        )

    def test_batch_broadcast_and_zero_inner(self):
        left = cuda.storage([1, 2, 3, 4, 5, 6], dtype="bfloat16")
        right = cuda.storage([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], dtype="float32")
        result = cuda.gemm(left, right, 2, 3, 2, batch=2, left_stride=0)
        self.assertValues(result.tolist(), [22, 28, 49, 64, 20, 14, 56, 41], 0)
        empty = cuda.storage([], dtype="bfloat16")
        target = cuda.full(6, 3, dtype="float32")
        cuda.gemm(empty, empty, 2, 0, 3, out=target, accumulate=True)
        self.assertValues(target.tolist(), [3] * 6, 0)
        cuda.gemm(empty, empty, 2, 0, 3, out=target)
        self.assertValues(target.tolist(), [0] * 6, 0)

    def test_autocast_and_fp64_compatibility(self):
        for dtype in ("float32", "float64"):
            a = cuda.storage([0.1, 0.2, 0.3, 0.4], dtype=dtype)
            with autocast():
                result = cuda.gemm(a, a, 2, 2, 2)
            values = (
                [bf16(x) for x in [0.1, 0.2, 0.3, 0.4]]
                if dtype == "float32"
                else [0.1, 0.2, 0.3, 0.4]
            )
            self.assertEqual(result.dtype, dtype)
            self.assertValues(
                result.tolist(),
                reference(values, values, 2, 2, 2),
                1e-7 if dtype == "float32" else 1e-15,
            )
        a = cuda.storage([0.1, 0.2, 0.3, 0.4], dtype="float32")
        result = cuda.gemm(a, a, 2, 2, 2, out_dtype="bfloat16")
        self.assertValues(
            result.tolist(),
            list(
                map(
                    bf16, reference([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 2, 2, 2)
                )
            ),
            0,
        )

    def test_invalid_output_and_alias(self):
        a = cuda.full(4, 1, dtype="float32")
        for kwargs in (
            {"out": a},
            {"accumulate": True},
            {"out": cuda.full(4, 0, dtype="bfloat16")},
            {"left_stride": -1},
        ):
            with self.assertRaises(ValueError):
                cuda.gemm(a, a, 2, 2, 2, **kwargs)
        with self.assertRaises(ValueError):
            cuda.gemm(a, a, 3, 2, 2)

    def test_large_mixed_transposes_and_partial_output_tiles(self):
        m, k, n = 1025, 129, 1025
        a = [math.sin(i * 0.013) * 0.03 for i in range(m * k)]
        b = [math.cos(i * 0.017) * 0.02 for i in range(k * n)]
        ra, rb = list(map(bf16, a)), list(map(bf16, b))
        left, right = (
            cuda.storage(a, dtype="bfloat16"),
            cuda.storage(b, dtype="float32"),
        )
        coordinates = (0, 15, 16, 63, 64, 127, 128, 1023, 1024)
        positions = [r * n + c for r in coordinates for c in coordinates]
        for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
            expected = [
                sum(
                    ra[p * m + r if ta else r * k + p]
                    * rb[c * k + p if tb else p * n + c]
                    for p in range(k)
                )
                for r in coordinates
                for c in coordinates
            ]
            target = cuda.full(m * n, 0.25, dtype="float32")
            result = cuda.gemm(
                left, right, m, k, n, transpose_left=ta, transpose_right=tb, out=target
            )
            self.assertIs(result, target)
            self.assertValues(cuda.gather(result, positions).tolist(), expected, 2e-7)
            cuda.gemm(
                left,
                right,
                m,
                k,
                n,
                transpose_left=ta,
                transpose_right=tb,
                out=target,
                accumulate=True,
            )
            self.assertValues(
                cuda.gather(target, positions).tolist(), [2 * x for x in expected], 4e-7
            )
        rounded = cuda.gemm(left, right, m, k, n, out_dtype="bfloat16")
        expected = [
            sum(ra[p + r * k] * rb[p * n + c] for p in range(k))
            for r in coordinates
            for c in coordinates
        ]
        self.assertEqual(rounded.nbytes, m * n * 2)
        self.assertValues(
            cuda.gather(rounded, positions).tolist(), list(map(bf16, expected)), 6.2e-5
        )

    def test_long_reduction_partial_tiles_batch_and_accumulate(self):
        m, k, n = 129, 8193, 1025
        left = cuda.full(m * k, 0.5, dtype="bfloat16")
        parts = (
            cuda.full(k * n, -0.25, dtype="float32"),
            cuda.full(k * n, 0.5, dtype="float32"),
        )
        right = cuda.concatenate(parts, ((k, n), (k, n)))
        target = cuda.full(2 * m * n, 0.25, dtype="float32")
        result = cuda.gemm(
            left, right, m, k, n, batch=2, left_stride=0, out=target, accumulate=True
        )
        self.assertIs(result, target)
        values = result.tolist()
        self.assertTrue(all(x == -k * 0.125 + 0.25 for x in values[: m * n]))
        self.assertTrue(all(x == k * 0.25 + 0.25 for x in values[m * n :]))


if __name__ == "__main__":
    unittest.main()
