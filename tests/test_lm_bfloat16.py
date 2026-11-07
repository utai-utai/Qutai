import unittest
from random import Random
from unittest.mock import patch

from qutai import nn, is_available
from qutai.backends import cpu, cpu_lm, cuda, cuda_lm
from qutai.lm.config import Config
from qutai.lm.decoder import Decoder
from qutai.lm.precision import autocast


@unittest.skipUnless(is_available("cuda"), "需要实际 CUDA/NVRTC")
class LMBFloat16Tests(unittest.TestCase):
    def assertClose(self, actual, expected, tolerance=0.012):
        a, b = actual.tolist(), expected.tolist()
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            self.assertLessEqual(abs(x - y), tolerance * max(1, abs(y)))

    def pair(self, values):
        gpu = cuda.storage(values, dtype="bfloat16")
        self.assertEqual(gpu.nbytes, gpu.size * 2)
        return gpu, cpu.storage(gpu.tolist(), dtype="float32")

    def test_norm_silu_rope_embedding_and_backward(self):
        random = Random(11)
        x, reference = self.pair([random.uniform(-1, 1) for _ in range(64)])
        gamma = cuda.storage([0.8 + i / 100 for i in range(8)], dtype="float32")
        gamma_cpu = cpu.storage(gamma.tolist(), dtype="float32")
        upstream = cuda.storage(
            [random.uniform(-0.2, 0.2) for _ in range(64)], dtype="float32"
        )
        upstream_cpu = cpu.storage(upstream.tolist(), dtype="float32")
        with patch.object(
            cuda.CUDAStorage,
            "tolist",
            side_effect=AssertionError("GPU训练核不能回读列表"),
        ):
            norm, inverse = cuda_lm.rms_norm(x, gamma, 8, 1e-6)
            dx, dw = cuda_lm.rms_norm_backward(upstream, x, gamma, inverse, 8)
            silu = cuda_lm.silu(x)
            silu_dx = cuda_lm.silu_backward(x, x)
            rope = cuda_lm.rope(x, 2, 2, 2, 8, 10000.0)
            restored = cuda_lm.rope(rope, 2, 2, 2, 8, 10000.0, inverse=True)
        norm_cpu, inverse_cpu = cpu_lm.rms_norm(reference, gamma_cpu, 8, 1e-6)
        dx_cpu, dw_cpu = cpu_lm.rms_norm_backward(
            upstream_cpu, reference, gamma_cpu, inverse_cpu, 8
        )
        self.assertEqual(
            (norm.dtype, inverse.dtype, dx.dtype, dw.dtype),
            ("float32", "float32", "bfloat16", "float32"),
        )
        self.assertClose(norm, norm_cpu, 2e-5)
        self.assertClose(dx, dx_cpu)
        self.assertClose(dw, dw_cpu, 2e-5)
        self.assertClose(silu, cpu_lm.silu(reference))
        self.assertClose(silu_dx, cpu_lm.silu_backward(reference, reference))
        self.assertClose(rope, cpu_lm.rope(reference, 2, 2, 2, 8, 10000.0))
        self.assertClose(restored, reference)
        weight = cuda.storage([i / 25 for i in range(40)], dtype="float32")
        embedded = cuda_lm.embedding(weight, (2, 0, 2), 8, out_dtype="bfloat16")
        self.assertEqual(embedded.nbytes, 48)
        gradient = cuda_lm.embedding_backward(embedded, (2, 0, 2), 5, 8)
        self.assertEqual(gradient.dtype, "float32")
        self.assertClose(
            gradient,
            cpu_lm.embedding_backward(
                cpu.storage(embedded.tolist(), dtype="float32"), (2, 0, 2), 5, 8
            ),
            1e-6,
        )

    def test_attention_is_bfloat16_with_fp32_lse_and_adjoint(self):
        random = Random(9)
        dims = (2, 5, 4, 2, 8)
        q, qr = self.pair([random.uniform(-0.4, 0.4) for _ in range(320)])
        k, kr = self.pair([random.uniform(-0.4, 0.4) for _ in range(160)])
        v, vr = self.pair([random.uniform(-0.4, 0.4) for _ in range(160)])
        g, gr = self.pair([random.uniform(-0.2, 0.2) for _ in range(320)])
        with (
            patch.object(
                cuda.CUDAStorage,
                "tolist",
                side_effect=AssertionError("Attention 不可读回列表"),
            ),
            patch.object(
                cpu_lm, "attention", side_effect=AssertionError("不可回退CPU")
            ),
        ):
            output, lse = cuda_lm.attention(q, k, v, *dims)
            gradients = cuda_lm.attention_backward(g, q, k, v, output, lse, *dims)
        expected, expected_lse = cpu_lm.attention(qr, kr, vr, *dims)
        self.assertEqual((output.dtype, lse.dtype), ("bfloat16", "float32"))
        self.assertEqual(output.nbytes, 640)
        self.assertClose(output, expected)
        self.assertClose(lse, expected_lse, 1e-5)
        # 反向使用真实保存的已舍入 BF16 输出，而非另一个未舍入状态。
        saved = cpu.storage(output.tolist(), dtype="float32")
        reference_gradients = cpu_lm.attention_backward(
            gr, qr, kr, vr, saved, expected_lse, *dims
        )
        for actual, reference in zip(gradients, reference_gradients, strict=True):
            self.assertEqual(actual.dtype, "bfloat16")
            self.assertClose(actual, reference)

    def test_d64_tiled_attention_partial_tiles_and_causal_mask(self):
        for time in (17, 33, 65):
            random = Random(time)
            dims = (1, time, 2, 1, 64)
            q = cuda.storage(
                [random.uniform(-0.4, 0.4) for _ in range(time * 128)], dtype="bfloat16"
            )
            k = cuda.storage(
                [random.uniform(-0.4, 0.4) for _ in range(time * 64)], dtype="bfloat16"
            )
            v = cuda.storage(
                [random.uniform(-0.4, 0.4) for _ in range(time * 64)], dtype="bfloat16"
            )
            actual, lse = cuda_lm.attention(q, k, v, *dims)
            expected, expected_lse = cuda_lm._attention_warp(q, k, v, *dims)
            self.assertClose(actual, expected, 0.004)
            self.assertClose(lse, expected_lse, 2e-5)
            changed = cuda.storage(
                v.tolist()[:64] + [100.0] * (v.size - 64), dtype="bfloat16"
            )
            causal, _ = cuda_lm.attention(q, k, changed, *dims)
            self.assertEqual(actual.tolist()[:128], causal.tolist()[:128])

    def test_router_fp32_and_bfloat16_expert_gather_combine(self):
        scores = [[3, 1, -2], [0, 2, 1], [1, -2, 4], [-1, 3, 2], [4, 0, 1]]
        flat_scores = [value for row in scores for value in row]
        values, reference = self.pair([0.03 * i - 0.2 for i in range(20)])
        gradient, reference_g = self.pair([0.05 * i - 0.4 for i in range(20)])
        with patch.object(
            cuda.CUDAStorage,
            "tolist",
            side_effect=AssertionError("专家数值不能读回主机"),
        ):
            route = cuda_lm.routing(cuda.storage(flat_scores, dtype="float32"), 5, 3, 2)
            outputs = tuple(
                (expert, cuda_lm.route_gather(values, route, expert, 4))
                for expert in range(3)
                if route["counts"][expert]
            )
            combined = cuda_lm.combine(outputs, route["weights"], route, 4)
            gradients = cuda_lm.combine_backward(
                gradient, outputs, route["weights"], route, 4
            )
        cpu_route = cpu_lm.routing(cpu.storage(flat_scores, dtype="float32"), 5, 3, 2)
        cpu_outputs = tuple(
            (expert, cpu_lm.route_gather(reference, cpu_route, expert, 4))
            for expert in range(3)
            if cpu_route["counts"][expert]
        )
        expected = cpu_lm.combine(cpu_outputs, cpu_route["weights"], cpu_route, 4)
        expected_gradients = cpu_lm.combine_backward(
            reference_g, cpu_outputs, cpu_route["weights"], cpu_route, 4
        )
        self.assertEqual(
            (route["weights"].dtype, route["aux"].dtype, combined.dtype),
            ("float32", "float32", "bfloat16"),
        )
        self.assertEqual(combined.nbytes, 40)
        self.assertClose(combined, expected)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            self.assertClose(actual, expected)
        self.assertEqual(gradients[-1].dtype, "float32")

    def test_head_rows_and_loss_reduce_fp32_keep_bfloat16_gradient(self):
        x, xr = self.pair([0.03 * i - 0.4 for i in range(64)])
        selected = cuda_lm.head_rows(x, 1, 4, 8, 4)
        self.assertEqual(selected.dtype, "bfloat16")
        self.assertClose(selected, cpu_lm.head_rows(xr, 1, 4, 8, 4), 1e-6)
        target = cuda.full(64, dtype="bfloat16")
        cuda_lm.head_rows_backward(target, selected, 1, 4, 8, 4)
        cpu_target = cpu.full(64, dtype="float32")
        cpu_lm.head_rows_backward(
            cpu_target, cpu_lm.head_rows(xr, 1, 4, 8, 4), 1, 4, 8, 4
        )
        self.assertClose(target, cpu_target, 1e-6)
        logits, lr = self.pair([0.02 * i - 0.3 for i in range(51)])
        labels = (3, -100, 11)
        scalar = cuda.storage([0.7], dtype="float32")
        with patch.object(
            cuda.CUDAStorage, "tolist", side_effect=AssertionError("loss核不能主机代算")
        ):
            loss = cuda_lm.head_loss(logits, labels, 3, 17)
            gradient = cuda_lm.head_gradient(logits, labels, 3, 17, scalar, 2)
        self.assertEqual(loss.dtype, "float32")
        self.assertIs(gradient, logits)
        self.assertEqual(gradient.dtype, "bfloat16")
        self.assertClose(loss, cpu_lm.head_loss(lr.copy(), labels, 3, 17), 1e-5)
        self.assertClose(
            gradient,
            cpu_lm.head_gradient(
                lr, labels, 3, 17, cpu.storage([0.7], dtype="float32"), 2
            ),
        )

    def test_dense_moe_real_bfloat16_activations_and_fp32_optimizer(self):
        for kind, width, heads, kv_heads, length in (
            ("dense", 32, 4, 2, 7),
            ("moe", 32, 4, 2, 7),
            ("dense", 128, 2, 1, 17),
            ("moe", 128, 2, 1, 17),
        ):
            config = Config(
                vocab_size=64,
                n_layers=2,
                d_model=width,
                n_heads=heads,
                n_kv_heads=kv_heads,
                d_ff=width * 2,
                max_seq_len=length,
                ffn_type=kind,
                n_experts=4,
                d_expert=width,
                top_k=2,
            )
            model = Decoder(config, device="cuda", seed=4)
            optimizer = nn.AdamW(model, learning_rate=0.001)
            observed = []
            original = cuda_lm.gemm

            def traced(*args, **kwargs):
                value = original(*args, **kwargs)
                observed.append((value.dtype, value.nbytes, value.size))
                return value

            for _ in range(2):
                optimizer.zero_grad()
                with (
                    patch.object(cuda_lm, "gemm", side_effect=traced),
                    patch.object(
                        cuda.CUDAStorage,
                        "tolist",
                        side_effect=AssertionError("训练中禁止读回列表"),
                    ),
                ):
                    with autocast():
                        loss = model.loss(
                            [
                                [i % 64 for i in range(1, length + 1)],
                                [(2 * i) % 64 for i in range(1, length + 1)],
                            ],
                            chunk_size=3,
                        )
                    loss.backward(retain_graph=False)
                self.assertEqual(loss.dtype, "float32")
                self.assertGreater(loss.item(), 0)
                nn.clip_grad_norm(model, 1.0)
                optimizer.step()
            self.assertTrue(
                any(
                    dtype == "bfloat16" and nbytes == size * 2
                    for dtype, nbytes, size in observed
                )
            )
            self.assertTrue(
                all(
                    parameter.dtype == "float32"
                    and (parameter.grad is None or parameter.grad.dtype == "float32")
                    for parameter in model.parameters()
                )
            )
            self.assertTrue(
                all(state.dtype == "float32" for state in optimizer.averages.values())
            )
            self.assertTrue(
                all(
                    state.dtype == "float32"
                    for state in optimizer.square_averages.values()
                )
            )


if __name__ == "__main__":
    unittest.main()
