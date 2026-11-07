import unittest
import weakref
import gc
from unittest.mock import patch

from qutai import Tensor, nn, is_available, lm
from qutai.core import grad
from qutai.core.context import set_recording
from qutai.backends import cpu, cuda
from qutai.lm.recompute import checkpoint
from qutai.lm.precision import autocast


class RecomputeTests(unittest.TestCase):
    def test_public_activation_checkpoint_is_callable(self):
        self.assertIs(lm.activation_checkpoint, checkpoint)
        x = Tensor(3.0, requires_grad=True)
        lm.activation_checkpoint(lambda value: value.square(), x).backward()
        self.assertEqual(x.grad.item(), 6)

    def test_unused_checkpoint_parameter_skips_adam_decay_and_counter(self):
        used = Tensor(2.0, requires_grad=True)
        unused = Tensor(3.0, requires_grad=True)
        optimizer = nn.AdamW([used, unused], learning_rate=0.1, weight_decay=0.1)
        (used + unused).square().backward()
        optimizer.step()
        before = (
            unused.item(),
            optimizer.averages[unused].tolist(),
            optimizer.square_averages[unused].tolist(),
        )
        optimizer.zero_grad()
        loss = checkpoint(lambda x: x * used, Tensor(2.0), parameters=(used, unused))
        loss.backward(retain_graph=False)
        self.assertIsNone(unused.grad)
        optimizer.step()
        self.assertEqual(optimizer.steps[used], 2)
        self.assertEqual(optimizer.steps[unused], 1)
        self.assertEqual(
            before,
            (
                unused.item(),
                optimizer.averages[unused].tolist(),
                optimizer.square_averages[unused].tolist(),
            ),
        )

    def test_checkpoint_rebuilds_only_during_backward_and_keeps_shared_parameters(self):
        model = nn.Model([nn.Linear(2, 3, seed=1), nn.ReLU(), nn.Linear(3, 1, seed=2)])
        x = Tensor([[0.2, -0.3], [0.7, 0.5]], requires_grad=True)
        expected = model(x).square().mean()
        expected.backward()
        weights = [p.grad.storage.tolist() for p in model.parameters()]
        xgrad = x.grad.storage.tolist()
        for p in model.parameters():
            p.zero_grad()
        x.zero_grad()
        calls = []

        def block(value):
            calls.append(1)
            return model(value)

        result = checkpoint(block, x, parameters=model.parameters())
        self.assertEqual(len(calls), 1)
        result.square().mean().backward()
        self.assertEqual(len(calls), 2)
        self.assertEqual(x.grad.storage.tolist(), xgrad)
        self.assertEqual([p.grad.storage.tolist() for p in model.parameters()], weights)
        result.square().mean().backward()
        self.assertEqual(len(calls), 3)
        for p, values in zip(model.parameters(), weights, strict=True):
            self.assertEqual(p.grad.storage.tolist(), [2 * v for v in values])

    def test_checkpoint_repeated_input_unused_input_and_declared_capture(self):
        x, unused, weight = (
            Tensor(2, requires_grad=True),
            Tensor(3, requires_grad=True),
            Tensor(4, requires_grad=True),
        )
        output = checkpoint(
            lambda a, b, c: (a + b) * weight, x, x, unused, parameters=(weight,)
        )
        output.backward()
        self.assertEqual(x.grad.item(), 8)
        self.assertIsNone(unused.grad)
        self.assertEqual(weight.grad.item(), 4)
        weight.zero_grad()
        checkpoint(lambda a: a * weight, weight, parameters=(weight,)).backward()
        self.assertEqual(weight.grad.item(), 8)
        with self.assertRaisesRegex(ValueError, "未声明"):
            checkpoint(lambda a: a * weight, x).backward()

    def test_release_graph_frees_saved_intermediates_and_rejects_reuse(self):
        x = Tensor(2, requires_grad=True)
        middle = x.square()
        reference = weakref.ref(middle)
        output = middle.square()
        del middle
        leaves = grad.backward(output, cpu.storage([1]), retain_graph=False)
        self.assertEqual(leaves[x].tolist(), [32])
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(output.parents, ())
        self.assertIsNone(output.backward_rule)
        with self.assertRaisesRegex(RuntimeError, "已释放"):
            grad.backward(output, cpu.storage([1]))
        self.assertEqual(
            grad.backward(x, cpu.storage([2]), retain_graph=False)[x].tolist(), [2]
        )
        first = x.square()
        second = first.square()
        reference = weakref.ref(second)
        saved_rule = first.backward_rule
        released_during_backward = []

        def observed_rule(incoming):
            released_during_backward.append(reference() is None)
            return saved_rule(incoming)

        first.backward_rule = observed_rule
        output = second.square()
        del first, second
        grad.backward(output, cpu.storage([1]), retain_graph=False)
        self.assertEqual(released_during_backward, [True])

    def test_checkpoint_forward_no_graph_and_backward_under_recording_pause(self):
        x = Tensor(3, requires_grad=True)
        with set_recording(False):
            self.assertFalse(checkpoint(lambda a: a.square(), x).requires_grad)
        output = checkpoint(lambda a: a.square(), x)
        with set_recording(False):
            output.backward()
        self.assertEqual(x.grad.item(), 6)

    def test_matmul_backward_computes_only_required_side(self):
        x, weight = Tensor([[1, 2]]), Tensor([[3], [4]], requires_grad=True)
        with patch.object(cpu, "transpose", wraps=cpu.transpose) as transposed:
            (x @ weight).sum().backward()
        self.assertEqual(transposed.call_count, 1)
        self.assertEqual(weight.grad.tolist(), [[1], [2]])

    def test_cuda_checkpoint_and_gemm_gradients_without_host_readout_or_transpose(self):
        if not is_available("cuda"):
            self.skipTest("需要 CUDA 实卡")
        models = [
            nn.Model([nn.Linear(3, 4, seed=1), nn.GELU(), nn.Linear(4, 2, seed=2)]).to(
                device, dtype="float32"
            )
            for device in ("cpu", "cuda")
        ]
        values = [[0.2, -0.4, 0.7], [-0.3, 0.8, 0.1]]
        expected = models[0](values).square().mean()
        expected.backward()
        x = Tensor(values, device="cuda", dtype="float32", requires_grad=True)
        with (
            patch.object(
                cuda.CUDAStorage,
                "tolist",
                side_effect=AssertionError("重计算不可回读主机"),
            ),
            patch.object(
                cuda,
                "transpose",
                side_effect=AssertionError("二维GEMM反向无需物理转置"),
            ),
        ):
            result = (
                checkpoint(lambda a: models[1](a), x, parameters=models[1].parameters())
                .square()
                .mean()
            )
            result.backward(retain_graph=False)
        self.assertAlmostEqual(result.item(), expected.item(), places=5)
        for actual, reference in zip(
            models[1].parameters(), models[0].parameters(), strict=True
        ):
            for a, b in zip(
                actual.grad.storage.tolist(),
                reference.grad.storage.tolist(),
                strict=True,
            ):
                self.assertAlmostEqual(a, b, places=5)
        with self.assertRaisesRegex(RuntimeError, "已释放"):
            result.backward()

    def test_matmul_backward_and_checkpoint_keep_forward_precision(self):
        if not is_available("cuda"):
            self.skipTest("需要 CUDA 实卡")

        def run(forward_bf16, backward_bf16, recompute=False):
            x = Tensor(
                [[0.12345, -0.34567, 0.56789], [0.98765, 0.45678, -0.23456]],
                device="cuda",
                dtype="float32",
                requires_grad=True,
            )
            weight = Tensor(
                [[0.43219, 0.54321], [-0.76543, 0.34567], [0.65432, -0.87654]],
                device="cuda",
                dtype="float32",
                requires_grad=True,
            )
            with autocast(forward_bf16):
                output = (
                    checkpoint(lambda a: a @ weight, x, parameters=(weight,))
                    if recompute
                    else x @ weight
                )
                loss = output.square().mean()
            with autocast(backward_bf16):
                loss.backward()
            return (
                output.storage.tolist(),
                x.grad.storage.tolist(),
                weight.grad.storage.tolist(),
            )

        self.assertEqual(run(True, False), run(True, True))
        self.assertEqual(run(False, True), run(False, False))
        self.assertEqual(run(True, False, recompute=True), run(True, True))
        self.assertNotEqual(run(True, False)[0], run(False, False)[0])


if __name__ == "__main__":
    unittest.main()
