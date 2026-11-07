import math
import unittest
from unittest.mock import patch

from qutai import is_available
from qutai.backends import cuda, cuda_lm


def warp_backward(g, q, k, v, out, lse, batch, time, heads, kv_heads, dim):
    dq = cuda.CUDAStorage(q.size, q.dtype)
    dk, dv = cuda.CUDAStorage(k.size, k.dtype), cuda.CUDAStorage(v.size, v.dtype)
    delta = cuda.CUDAStorage(lse.size, "float64" if q.dtype == "float64" else "float32")
    cuda_lm._run("lm_attention_delta", lse.size, cuda._p(g), cuda._p(out), cuda._p(delta),
                 cuda._i(lse.size), cuda._i(dim), dtype=q.dtype, block=True, block_size=32)
    dims = tuple(cuda._i(x) for x in (batch, time, heads, kv_heads, dim))
    arguments = (cuda._p(g), cuda._p(q), cuda._p(k), cuda._p(v), cuda._p(lse), cuda._p(delta))
    cuda_lm._run("lm_attention_backward_q", lse.size, *arguments, cuda._p(dq), *dims,
                 dtype=q.dtype, block=True, block_size=32)
    cuda_lm._run("lm_attention_backward_kv", batch * time * kv_heads, *arguments,
                 cuda._p(dk), cuda._p(dv), *dims, dtype=q.dtype, block=True, block_size=32)
    return dq, dk, dv


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class TiledAttentionTests(unittest.TestCase):
    def make(self, dims, dtype="bfloat16", std=0.5):
        batch, time, heads, kv_heads, dim = dims
        q = cuda.normal(batch * time * heads * dim, std=std, seed=41)
        k = cuda.normal(batch * time * kv_heads * dim, std=std, seed=42)
        v = cuda.normal(k.size, std=0.5, seed=43)
        g = cuda.normal(q.size, std=0.5, seed=44)
        if dtype != "float32":
            q, k, v, g = (cuda.from_storage(x, dtype=dtype) for x in (q, k, v, g))
        return q, k, v, g

    def test_gqa_remainders_match_warp_gradient(self):
        for dims in ((1, 17, 2, 1, 64), (2, 33, 4, 2, 64), (1, 65, 4, 1, 64), (1, 128, 4, 2, 64)):
            for std in (0.5, 1.0):
                with self.subTest(dims=dims, std=std):
                    q, k, v, g = self.make(dims, std=std)
                    out, lse = cuda_lm.attention(q, k, v, *dims)
                    reference = warp_backward(g, q, k, v, out, lse, *dims)
                    actual = cuda_lm.attention_backward(g, q, k, v, out, lse, *dims)
                    for x, y in zip(actual, reference, strict=True):
                        self.assertEqual(x.dtype, "bfloat16")
                        self.assertEqual(x.nbytes, x.size * 2)
                        values, expected = x.tolist(), y.tolist()
                        self.assertTrue(all(map(math.isfinite, values)))
                        squared = sum((a - b) ** 2 for a, b in zip(values, expected))
                        norm = sum(b * b for b in expected)
                        self.assertLess(math.sqrt(squared / max(norm, 1e-30)), 0.015)

    def test_backward_is_bitwise_repeatable(self):
        dims = (2, 65, 4, 1, 64)
        q, k, v, g = self.make(dims)
        out, lse = cuda_lm.attention(q, k, v, *dims)
        expected = None
        for _ in range(4):
            values = tuple(x.tolist() for x in cuda_lm.attention_backward(g, q, k, v, out, lse, *dims))
            if expected is None:
                expected = values
            else:
                self.assertEqual(values, expected)

    def test_large_bfloat16_dispatches_to_tensor_core_backward(self):
        dims = (2, 65, 4, 1, 64)
        q, k, v, g = self.make(dims)
        out, lse = cuda_lm.attention(q, k, v, *dims)
        runtime = cuda.get_runtime()
        with patch.object(runtime, "launch", wraps=runtime.launch) as launch:
            cuda_lm.attention_backward(g, q, k, v, out, lse, *dims)
        names = [call.args[1] for call in launch.call_args_list]
        self.assertIn("lm_attention_bf16_dq", names)
        self.assertIn("lm_attention_bf16_dkv", names)
        self.assertNotIn("lm_attention_backward_q", names)

    def test_future_tokens_do_not_receive_first_output_gradient(self):
        dims = (1, 33, 2, 1, 64)
        q, k, v, _ = self.make(dims)
        g = cuda.storage([0.5] * 128 + [0] * (q.size - 128), dtype="bfloat16")
        out, lse = cuda_lm.attention(q, k, v, *dims)
        dq, dk, dv = cuda_lm.attention_backward(g, q, k, v, out, lse, *dims)
        self.assertEqual(dq.tolist()[128:], [0] * (dq.size - 128))
        self.assertEqual(dk.tolist()[64:], [0] * (dk.size - 64))
        self.assertEqual(dv.tolist()[64:], [0] * (dv.size - 64))

    def test_other_precisions_keep_generic_backward(self):
        dims = (1, 7, 2, 1, 8)
        for dtype in ("float32", "float64"):
            q, k, v, g = self.make(dims, dtype=dtype)
            out, lse = cuda_lm.attention(q, k, v, *dims)
            expected = warp_backward(g, q, k, v, out, lse, *dims)
            actual = cuda_lm.attention_backward(g, q, k, v, out, lse, *dims)
            self.assertEqual(tuple(x.tolist() for x in actual), tuple(x.tolist() for x in expected))
