import unittest
from array import array
from math import sqrt
from unittest.mock import patch

from qutai import Tensor, nn, qnn
from qutai.backends import cpu
from qutai.backends.storage import Storage
from qutai.quantum import CNOT, H, RY, RZ, S, X, Y, Z, Circuit, State


def scalar_export_only(data):
    """门角度可以读取单个实数；振幅和计算向量应留在后端。"""
    if data.dtype != "float64" or data.size != 1:
        raise AssertionError("计算过程不应导出整个数值向量")
    return [data[0]]


class QuantumStorageTests(unittest.TestCase):
    def test_state_owns_contiguous_complex_storage_and_copy_is_independent(self):
        amplitudes = [1 / sqrt(2), 0j, 0j, 1j / sqrt(2)]
        state = State(2, amplitudes)
        self.assertIsInstance(state.storage, Storage)
        self.assertIsInstance(state.storage.buffer, array)
        self.assertEqual((state.device, state.dtype, state.storage.size), ("cpu", "complex128", 4))
        self.assertEqual(len(state.storage.buffer), 8)
        self.assertIs(state.amplitudes, state.storage)
        amplitudes[0] = 0j
        self.assertAlmostEqual(abs(state.amplitudes[0]), 1 / sqrt(2))
        copied = state.copy()
        copied.amplitudes[0] = 0j
        self.assertNotEqual(copied.amplitudes, state.amplitudes)
        self.assertIsNot(copied.storage, state.storage)
        state.amplitudes = [0j, 1j, 0j, 0j]
        self.assertEqual(state.dtype, "complex128")
        self.assertEqual(state.probabilities(), [0, 1, 0, 0])

    def test_observable_accepts_unnormalized_storage_without_export_or_state_wrapper(self):
        amplitudes = cpu.storage([2, 3j], dtype="complex128")
        operator = 2 * Y(0) + Z(0) - 0.5
        with patch.object(cpu.CPUStorage, "tolist", autospec=True, side_effect=scalar_export_only):
            result = operator.apply(amplitudes, 1)
        self.assertIsInstance(result, Storage)
        self.assertEqual(result.dtype, "complex128")
        self.assertEqual(result.tolist(), [7 + 0j, -0.5j])
        self.assertEqual(amplitudes.tolist(), [2 + 0j, 3j])

    def test_quantum_gradient_paths_keep_complex_and_real_vectors_in_storage(self):
        results = []
        for method in ("adjoint", "parameter-shift"):
            parameter = Tensor(0.31, requires_grad=True)
            initial = State(3, [1 / sqrt(2), 0, 0, 0, 0, 0, 0, 1j / sqrt(2)])
            with patch.object(cpu.CPUStorage, "tolist", autospec=True, side_effect=scalar_export_only):
                angle = parameter * 2 + 0.1
                circuit = Circuit(3, [H(1), RY(0, angle), S(2), CNOT(0, 2), RZ(1, parameter)])
                features = circuit.expectation([Z(0), X(1) @ Y(2), Z(0) @ Z(2)], initial, method=method)
                probabilities = circuit.probabilities((2, 0), initial, method=method)
                loss = features.square().sum() + probabilities.square().sum()
                loss.backward()
            self.assertIsInstance(features.storage, Storage)
            self.assertIsInstance(probabilities.storage, Storage)
            self.assertIsInstance(parameter.grad.storage, Storage)
            self.assertEqual(parameter.grad.dtype, "float64")
            results.append((loss.item(), parameter.grad.item()))
        self.assertAlmostEqual(results[0][0], results[1][0], places=12)
        self.assertAlmostEqual(results[0][1], results[1][1], places=12)

    def test_hybrid_network_trains_without_exporting_intermediate_vectors(self):
        model = nn.Model([nn.Linear(2, 2, seed=7), qnn.PQC(2, depth=2, seed=8), nn.Linear(2, 1, seed=9)])
        inputs, targets = [[0.2, -0.4], [0.7, 0.1]], [[0.3], [-0.2]]
        loss_function, optimizer = nn.MSE(), nn.AdamW(model, learning_rate=0.01)
        before = [parameter.storage.copy() for parameter in model.parameters()]
        with patch.object(cpu.CPUStorage, "tolist", autospec=True, side_effect=scalar_export_only):
            for _ in range(2):
                optimizer.zero_grad()
                loss = loss_function(model(inputs), targets)
                loss.backward()
                optimizer.step()
        for parameter, previous in zip(model.parameters(), before, strict=True):
            self.assertIsInstance(parameter.storage, Storage)
            self.assertIsInstance(parameter.grad.storage, Storage)
            self.assertNotEqual(parameter.storage, previous)
        self.assertEqual(model(inputs, train=False).shape, (2, 1))

    def test_plain_state_and_circuit_readouts_keep_their_public_numeric_types(self):
        circuit = Circuit(2, [H(0), CNOT(0, 1)])
        state = circuit.run()
        self.assertIsInstance(state.expectation(Z(0)), float)
        self.assertIsInstance(state.expectation([Z(0), Z(1)]), list)
        self.assertIsInstance(circuit.expectation([Z(0), Z(1)]), list)
        self.assertIsInstance(circuit.probabilities(), list)
        probabilities = state.probabilities()
        self.assertIsInstance(probabilities, list)
        self.assertTrue(all(type(value) is float for value in probabilities))
        before = state.storage.copy()
        counts = state.counts(20, seed=42)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(state.storage, before)
        measured = state.measure(seed=7)
        self.assertIn(measured, ("00", "11"))
        self.assertAlmostEqual(sum(state.probabilities()), 1)
        self.assertEqual(state.dtype, "complex128")

    def test_unimplemented_state_device_is_rejected_before_allocating(self):
        with self.assertRaisesRegex(ValueError, "设备"):
            State(1, device="bad")


if __name__ == "__main__":
    unittest.main()
