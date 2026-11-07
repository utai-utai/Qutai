import unittest
from unittest.mock import patch

from qutai import Tensor, is_available, qnn, nn
from qutai.backends import cuda, cuda_quantum
from qutai.quantum.circuit import Circuit


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class QuantumBatchTests(unittest.TestCase):
    def assertClose(self, left, right, tolerance=4e-5):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            self.assertLess(abs(a - b), tolerance * max(1, abs(b)))

    def test_sequence_forward_and_shared_weight_adjoint_match_cpu(self):
        for dtype in ("float32", "float64"):
            for entangler in ("cnot", "cz"):
                with self.subTest(dtype=dtype, entangler=entangler):
                    inputs = [0.03 * i - 0.4 for i in range(36)]
                    reference = qnn.PQC(3, depth=2, entangler=entangler, seed=7).to(
                        dtype=dtype
                    )
                    layer = qnn.PQC(3, depth=2, entangler=entangler, seed=7).to(
                        "cuda", dtype=dtype
                    )
                    x = Tensor(inputs, shape=(2, 6, 3), dtype=dtype, requires_grad=True)
                    y = Tensor(
                        inputs,
                        shape=(2, 6, 3),
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )
                    expected = reference(x)
                    expected.square().mean().backward()
                    with (
                        patch.object(
                            Circuit,
                            "expectation",
                            side_effect=AssertionError("不能逐token执行Circuit"),
                        ),
                        patch.object(
                            cuda.CUDAStorage,
                            "tolist",
                            side_effect=AssertionError("训练必须留在GPU"),
                        ),
                    ):
                        result = layer(y)
                        result.square().mean().backward()
                    self.assertEqual(result.shape, (2, 6, 3))
                    self.assertClose(result.storage.tolist(), expected.storage.tolist())
                    self.assertClose(y.grad.storage.tolist(), x.grad.storage.tolist())
                    self.assertClose(
                        layer.weight.grad.storage.tolist(),
                        reference.weight.grad.storage.tolist(),
                    )

    def test_chunking_forward_backward_and_masked_rows(self):
        layer = qnn.PQC(4, depth=2, seed=3).to("cuda", dtype="float32")
        inputs = cuda.storage([0.02 * i for i in range(44)], dtype="float32")
        gradient = cuda.storage(
            [1.0 if row < 7 else 0.0 for row in range(11) for _ in range(4)],
            dtype="float32",
        )
        reference = cuda_quantum.pqc_forward(
            inputs,
            layer.weight.storage,
            layer.rotations,
            layer.entangler,
            chunk_size=11,
        )
        dx, dw = cuda_quantum.pqc_backward(
            gradient,
            inputs,
            layer.weight.storage,
            layer.rotations,
            layer.entangler,
            chunk_size=11,
        )
        for chunk in (1, 3, 5):
            result = cuda_quantum.pqc_forward(
                inputs,
                layer.weight.storage,
                layer.rotations,
                layer.entangler,
                chunk_size=chunk,
            )
            xg, wg = cuda_quantum.pqc_backward(
                gradient,
                inputs,
                layer.weight.storage,
                layer.rotations,
                layer.entangler,
                chunk_size=chunk,
            )
            self.assertClose(result.tolist(), reference.tolist())
            self.assertClose(xg.tolist(), dx.tolist())
            self.assertClose(wg.tolist(), dw.tolist())
            self.assertEqual(xg.tolist()[28:], [0.0] * 16)

    def test_two_adamw_updates_and_forward_parameter_snapshot(self):
        models = [
            qnn.PQC(2, depth=2, seed=13).to(device, dtype="float32")
            for device in ("cpu", "cuda")
        ]
        for model in models:
            optimizer = nn.AdamW(model, learning_rate=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                nn.MSE()(
                    model([[0.1, -0.2], [0.7, 0.4]]), [[0.3, 0.5], [-0.1, 0.2]]
                ).backward()
                optimizer.step()
        self.assertClose(
            models[0].weight.storage.tolist(), models[1].weight.storage.tolist()
        )
        layer = models[1]
        y = layer([[0.2, 0.3]])
        saved = layer.weight.storage.copy()
        cuda.copy_into(
            layer.weight.storage, cuda.full(saved.size, 0.7, dtype="float32")
        )
        layer.weight.zero_grad()
        y.sum().backward()
        actual = layer.weight.grad.storage.tolist()
        cuda.copy_into(layer.weight.storage, saved)
        layer.weight.zero_grad()
        layer([[0.2, 0.3]]).sum().backward()
        self.assertClose(actual, layer.weight.grad.storage.tolist())


if __name__ == "__main__":
    unittest.main()
