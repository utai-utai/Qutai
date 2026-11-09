import gc
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qutai import Tensor, nn
from qutai.backends import cpu_lm, cuda, cuda_lm
from qutai.backends.device import get_backend
from qutai.backends.precision import matmul_precision
from qutai.lm.config import Config
from qutai.lm.decoder import Decoder
from qutai.lm.layers import projection_loss
from qutai.lm.precision import autocast


def tiny(**changes):
    values = dict(
        vocab_size=17,
        n_layers=2,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=12,
        max_seq_len=8,
        n_experts=3,
        top_k=2,
        d_expert=6,
    )
    values.update(changes)
    return Config(**values)


def copy_parameters(source, target):
    for p, q in zip(source.parameters(), target.parameters(), strict=True):
        get_backend(q.device).copy_into(
            q.storage, get_backend(q.device).storage(p.storage.tolist(), dtype=q.dtype)
        )


class DecoderTests(unittest.TestCase):
    def assertClose(self, a, b, tolerance=2e-6):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            self.assertLessEqual(abs(x - y), tolerance * (1 + abs(y)))

    def test_real_configuration_parameter_counts_without_allocation(self):
        self.assertEqual(Config().parameter_counts()["total_parameters"], 306226176)
        self.assertEqual(
            Config(ffn_type="moe").parameter_counts(),
            dict(total_parameters=966992896, active_parameters=306390016),
        )
        model = Decoder(tiny(), seed=7)
        self.assertEqual(
            sum(p.size for p in model.parameters()),
            model.parameter_counts()["total_parameters"],
        )
        self.assertEqual(len(dict(model.named_parameters())), len(model.parameters()))

    def test_configuration_rejects_non_integer_or_nonfinite_values(self):
        for changes in (
            {"d_ff": 8.5},
            {"vocab_size": True},
            {"top_k": 1.5},
            {"rope_theta": float("nan")},
            {"norm_eps": float("inf")},
            {"aux_loss_coef": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny(**changes)

    def test_dense_gradients_and_chunked_loss(self):
        model = Decoder(
            tiny(n_layers=1, activation_checkpointing=False), dtype="float64", seed=7
        )
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        labels = [[1, -100, 3, 4], [2, 3, 4, 5]]
        loss = model.loss(tokens, labels, chunk_size=2)
        loss.backward()
        selected = [
            (model.embedding.weight, 3),
            (model.blocks[0].attn.q.weight, 4),
            (model.blocks[0].ffn.gate.weight, 8),
        ]
        for parameter, index in selected:
            original = parameter.storage[index]
            parameter.storage[index] = original + 1e-5
            upper = model.loss(tokens, labels, train=False, chunk_size=20).item()
            parameter.storage[index] = original - 1e-5
            lower = model.loss(tokens, labels, train=False, chunk_size=20).item()
            parameter.storage[index] = original
            self.assertAlmostEqual(
                parameter.grad.storage[index], (upper - lower) / 2e-5, places=7
            )
        self.assertAlmostEqual(
            loss.item(),
            model.loss(tokens, labels, train=False, chunk_size=20).item(),
            places=12,
        )

    def test_checkpoint_parity_and_causal_attention(self):
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        plain = Decoder(tiny(activation_checkpointing=False), dtype="float64", seed=2)
        checked = Decoder(tiny(activation_checkpointing=True), dtype="float64", seed=2)
        plain.loss(tokens).backward()
        checked.loss(tokens).backward()
        for a, b in zip(plain.parameters(), checked.parameters()):
            self.assertClose(a.grad.storage.tolist(), b.grad.storage.tolist(), 1e-10)
        first = plain([[1, 2, 3, 4]], train=False).tolist()[0]
        second = plain([[1, 2, 9, 8]], train=False).tolist()[0]
        self.assertClose(first[0] + first[1], second[0] + second[1], 1e-12)

    def test_masked_targets_and_linear_without_bias_roundtrip(self):
        model = Decoder(tiny(n_layers=1), seed=2)
        loss = model.loss([[1, 2, 3]], labels=[[-100, -100, -100]])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for p in model.parameters():
            self.assertTrue(all(v == 0 for v in p.grad.storage))
        with TemporaryDirectory() as folder:
            network = nn.Model([nn.Linear(3, 2, bias=False, seed=1)])
            path = Path(folder) / "model.json"
            network.save(path)
            restored = nn.Model().load(path)
            self.assertIsNone(restored.layers[0].bias)
            self.assertEqual(
                network([[1, 2, 3]]).tolist(), restored([[1, 2, 3]]).tolist()
            )


@unittest.skipUnless(cuda.is_available(), "需要真实 CUDA 设备")
class CUDADecoderTests(unittest.TestCase):
    assertClose = DecoderTests.assertClose

    def test_gpu_forward_gradients_two_steps_and_no_cpu_fallback(self):
        cpu_model = Decoder(tiny(), dtype="float32", seed=7)
        gpu_model = Decoder(tiny(), device="cuda", seed=8)
        copy_parameters(cpu_model, gpu_model)
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        cpu_opt = nn.AdamW(cpu_model, learning_rate=0.001)
        gpu_opt = nn.AdamW(gpu_model, learning_rate=0.001)
        for _ in range(2):
            cpu_opt.zero_grad()
            gpu_opt.zero_grad()
            expected = cpu_model.loss(tokens, chunk_size=3)
            expected.backward()
            with (
                patch.object(
                    cuda.CUDAStorage,
                    "tolist",
                    side_effect=AssertionError("训练不能读取显存列表"),
                ),
                patch.object(
                    cpu_lm, "gemm", side_effect=AssertionError("不能 CPU 回退")
                ),
            ):
                actual = gpu_model.loss(tokens, chunk_size=3)
                actual.backward()
                gpu_opt.step()
            cpu_opt.step()
            self.assertAlmostEqual(actual.item(), expected.item(), places=5)
            for a, b in zip(cpu_model.parameters(), gpu_model.parameters()):
                self.assertClose(a.grad.storage.tolist(), b.grad.storage.tolist(), 2e-5)
                self.assertClose(a.storage.tolist(), b.storage.tolist(), 2e-5)

    def test_autocast_forward_precision_is_restored_during_backward(self):
        model = Decoder(tiny(activation_checkpointing=False), device="cuda", seed=7)
        calls = []
        original = cuda_lm.gemm

        def capture(*args, **kwargs):
            calls.append(matmul_precision())
            return original(*args, **kwargs)

        with patch.object(cuda_lm, "gemm", side_effect=capture):
            with autocast():
                loss = model.loss([[1, 2, 3, 4]])
            calls.clear()
            loss.backward()
        self.assertTrue(calls)
        self.assertTrue(all(value == "bfloat16" for value in calls))

    def test_projection_head_peak_is_bounded_by_chunk(self):
        width, vocab, time = 32, 4096, 129
        hidden = Tensor(
            cuda.normal(time * width, seed=1),
            shape=(1, time, width),
            requires_grad=True,
        )
        weight = Tensor(
            cuda.normal(vocab * width, seed=2), shape=(vocab, width), requires_grad=True
        )
        labels = [list(range(time))]
        peaks = []
        for chunk in (16, 128):
            gc.collect()
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            baseline = cuda.memory_stats()["allocated_bytes"]
            loss = projection_loss(hidden, weight, labels, chunk)
            cuda.synchronize()
            peaks.append(cuda.memory_stats()["peak_allocated_bytes"] - baseline)
            self.assertTrue(math.isfinite(loss.item()))
            del loss
        self.assertLess(peaks[0], peaks[1] / 2)

    def test_repeated_embedding_indices_have_fixed_accumulation_order(self):
        width, count = 17, 41
        ids = [(row * 11) % 37 for row in range(1537)]
        values = [
            10000.0 if i % 3 == 0 else -9999.999 if i % 3 == 1 else 0.00031
            for i in range(len(ids) * width)
        ]
        gradient = cuda.storage(values, dtype="float32")
        reference = cpu_lm.embedding_backward(
            cpu_lm.cpu.storage(values, dtype="float32"), ids, count, width
        ).tolist()
        for _ in range(4):
            with patch.object(
                cuda.CUDAStorage,
                "tolist",
                side_effect=AssertionError("不能在 CPU 累加梯度"),
            ):
                actual = cuda_lm.embedding_backward(gradient, ids, count, width)
            self.assertEqual(actual.tolist(), reference)

    def test_grouped_attention_backward_is_bitwise_repeatable(self):
        dims = (2, 17, 4, 2, 8)
        q = cuda.normal(2 * 17 * 4 * 8, seed=11)
        k = cuda.normal(2 * 17 * 2 * 8, seed=12)
        v = cuda.normal(k.size, seed=13)
        incoming = cuda.normal(q.size, seed=14)
        output, lse = cuda_lm.attention(q, k, v, *dims)
        reference = None
        for _ in range(4):
            gradients = cuda_lm.attention_backward(
                incoming, q, k, v, output, lse, *dims
            )
            values = tuple(g.tolist() for g in gradients)
            if reference is None:
                reference = values
            else:
                self.assertEqual(values, reference)
