"""CUDA 实卡量子前向、局部反向和两步混合训练；不能用 CPU 代算。"""

import unittest
from math import cos, sin, sqrt
from unittest.mock import patch

from qutai import Tensor, is_available, nn, qnn, quantum as q
from qutai.backends import cpu, cuda
from qutai.backends.cuda_runtime import get_runtime


@unittest.skipUnless(is_available("cuda"), "需要实际可用的 CUDA 设备与 NVRTC")
class CUDAQuantumTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, tolerance=3e-5):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            self.assertLessEqual(abs(a - b), tolerance * max(1, abs(b)))

    def test_inner_readout_and_basis_dtype_boundaries(self):
        for dtype in ("float32", "float64", "complex64", "complex128"):
            values = [1 + 2j, -3j] if dtype.startswith("complex") else [2, -3]
            actual = cuda.storage(values, dtype=dtype)
            expected = cpu.storage(values, dtype=dtype)
            self.assertAlmostEqual(cuda.inner(actual, actual), cpu.inner(expected, expected), places=5)
        indices = cuda.storage([1, 2], dtype="int64")
        with self.assertRaises(ValueError): cuda.inner(indices, indices)
        for dtype in ("float32", "float64", "int64"):
            with self.assertRaises(ValueError): cuda.basis_state(1, dtype=dtype)

    def test_bell_measurement_and_reordered_marginals(self):
        for dtype in ("complex64", "complex128"):
            with self.subTest(dtype=dtype):
                bell = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run(q.State(2, device="cuda", dtype=dtype))
                self.assertVectorClose(bell.probabilities(), [0.5, 0, 0, 0.5])
                self.assertVectorClose(bell.probabilities((1, 0)), [0.5, 0, 0, 0.5])
                self.assertVectorClose(bell.probabilities(0), [0.5, 0.5])
                self.assertAlmostEqual(bell.expectation(q.Z(0) @ q.Z(1)), 1, places=5)
                original = bell.storage.tolist()
                self.assertTrue(set(bell.sample(32, seed=42)) <= {"00", "11"})
                self.assertEqual(bell.storage.tolist(), original)
                outcome = bell.measure(seed=42)
                expected = [0, 0, 0, 0]
                expected[int(outcome, 2)] = 1
                self.assertVectorClose(bell.probabilities(), expected)

    def test_local_complex_gates_match_cpu_on_nonadjacent_wires(self):
        values = [complex(i - 3, 2*i % 5 - 2) for i in range(8)]
        norm = sqrt(sum(abs(value) ** 2 for value in values))
        values = [value / norm for value in values]
        gates = [q.S(1), q.RX(0, 0.37), q.CNOT(2, 0), q.RY(1, -0.51), q.RZ(2, 0.73), q.SWAP(2, 0)]
        for dtype in ("complex64", "complex128"):
            with self.subTest(dtype=dtype):
                expected = q.Circuit(3, gates).run(q.State(3, values, dtype=dtype))
                actual = q.Circuit(3, gates).run(q.State(3, values, device="cuda", dtype=dtype))
                self.assertVectorClose(actual.storage.tolist(), expected.storage.tolist())
                self.assertVectorClose(actual.probabilities((2, 0)), expected.probabilities((2, 0)))

    def test_gpu_rotation_values_and_gradients_never_read_angles_on_cpu(self):
        cases = [(q.RX, [], q.Y(0), -sin(0.4), -cos(0.4)),
                 (q.RY, [], q.Z(0), cos(0.4), -sin(0.4)),
                 (q.RZ, [q.H(0)], q.Y(0), sin(0.4), cos(0.4))]
        launches = get_runtime().launches
        for dtype in ("float32", "float64"):
            for method in ("adjoint", "parameter-shift"):
                for rotation, before, observable, expected, derivative in cases:
                    with self.subTest(dtype=dtype, method=method, rotation=rotation.__name__):
                        angle = Tensor(0.4, device="cuda", dtype=dtype, requires_grad=True)
                        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("量子训练不能读回主机列表")), \
                             patch.object(cpu, "apply_gate", side_effect=AssertionError("量子门不能回退 CPU")):
                            result = q.Circuit(1, [*before, rotation(0, angle)]).expectation(observable, method=method)
                            result.backward()
                        self.assertEqual(result.device, "cuda")
                        self.assertEqual(angle.grad.dtype, dtype)
                        self.assertAlmostEqual(result.item(), expected, places=5)
                        self.assertAlmostEqual(angle.grad.item(), derivative, places=5)
        self.assertGreater(get_runtime().launches, launches)

    def test_probability_gradients_and_shared_forward_snapshot(self):
        for method in ("adjoint", "parameter-shift"):
            angle = Tensor(0.37, device="cuda", dtype="float32", requires_grad=True)
            with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("量子反向不能读回列表")):
                circuit = q.Circuit(2, [q.RY(0, angle), q.CNOT(0, 1), q.RZ(1, angle)])
                result = circuit.probabilities(wires=(1, 0), method=method)
                loss = nn.MSE()(result, [0.15, 0, 0, 0.85])
                cuda.copy_into(angle.storage, cuda.storage([1.2], dtype="float32"))
                loss.backward()
            reference = Tensor(0.37, dtype="float32", requires_grad=True)
            circuit = q.Circuit(2, [q.RY(0, reference), q.CNOT(0, 1), q.RZ(1, reference)])
            expected = nn.MSE()(circuit.probabilities(wires=(1, 0), method=method), [0.15, 0, 0, 0.85])
            expected.backward()
            self.assertAlmostEqual(loss.item(), expected.item(), places=5)
            self.assertAlmostEqual(angle.grad.item(), reference.grad.item(), places=5)

    def test_two_updates_for_hybrid_pqc_and_qaoa_match_cpu(self):
        for kind in ("hybrid", "qaoa"):
            def make(device):
                if kind == "hybrid":
                    return nn.Model([nn.Linear(2, 2, seed=1), qnn.PQC(2, depth=1, seed=2), nn.Linear(2, 1, seed=3)]).to(device, dtype="float32")
                return q.QAOA(4, q.maxcut([(0, 1), (1, 2), (2, 3), (3, 0)]), depth=1, seed=42).to(device, dtype="float32")

            expected, actual = make("cpu"), make("cuda")
            for model in (expected, actual):
                optimizer = nn.SGD(model, learning_rate=0.05)
                device = model.parameters()[0].device
                for _ in range(2):
                    optimizer.zero_grad()
                    if device == "cuda":
                        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("训练参数和状态必须保留在 GPU")), \
                             patch.object(cpu, "apply_gate", side_effect=AssertionError("量子门不能回退 CPU")):
                            loss = nn.MSE()(model([[0.2, -0.7], [0.8, 0.1]]), [[0.3], [-0.2]]) if kind == "hybrid" else model.expectation()
                            loss.backward()
                            optimizer.step()
                    else:
                        loss = nn.MSE()(model([[0.2, -0.7], [0.8, 0.1]]), [[0.3], [-0.2]]) if kind == "hybrid" else model.expectation()
                        loss.backward()
                        optimizer.step()
            with self.subTest(kind=kind):
                for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
                    self.assertVectorClose(a.storage.tolist(), b.storage.tolist())


if __name__ == "__main__":
    unittest.main()
