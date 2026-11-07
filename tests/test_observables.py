import unittest
from math import sqrt

from qutai import Tensor
from qutai.quantum import Circuit, CNOT, Gate, H, Hermitian, Observable, RY, S, SWAP, X, Y, Z
from qutai.quantum.observables import as_observable


def expectation(amplitudes, qubits, observable):
    transformed = as_observable(observable).apply(amplitudes, qubits)
    return sum(value.conjugate() * result for value, result in zip(amplitudes, transformed, strict=True)).real


class ObservableTests(unittest.TestCase):
    def test_bell_pauli_correlations(self):
        amplitudes = [1 / sqrt(2), 0j, 0j, 1 / sqrt(2)]
        for observable, expected in ((X(0) @ X(1), 1), (Y(0) @ Y(1), -1), (Z(0) @ Z(1), 1), (Z(0) @ Y(1), 0)):
            self.assertAlmostEqual(expectation(amplitudes, 2, observable), expected, places=12)

    def test_weighted_sums_constants_and_builtin_sum(self):
        amplitudes = [1 + 0j, 0j]
        hamiltonian = 0.5 * Z(0) + 0.3 * X(0)
        self.assertIsInstance(hamiltonian, Observable)
        self.assertAlmostEqual(expectation(amplitudes, 1, hamiltonian), 0.5)
        self.assertAlmostEqual(expectation(amplitudes, 1, 2 - hamiltonian), 1.5)
        self.assertAlmostEqual(expectation(amplitudes, 1, (hamiltonian - 1) / 2), -0.25)
        self.assertAlmostEqual(expectation(amplitudes, 1, sum([Z(0), -Z(0), 3 * Z(0)])), 3)

    def test_tensor_product_distributes_sums_without_full_matrix(self):
        amplitudes = Circuit(2, [H(0), RY(1, 0.6), CNOT(1, 0)]).run().amplitudes
        product = (X(0) + Z(0)) @ (X(1) - Z(1))
        expanded = X(0) @ X(1) - X(0) @ Z(1) + Z(0) @ X(1) - Z(0) @ Z(1)
        self.assertEqual(product.apply(amplitudes, 2), expanded.apply(amplitudes, 2))
        self.assertEqual(len(product.terms), 4)
        self.assertTrue(all(len(matrix) == 2 for _, factors in product.terms for _, matrix in factors))

    def test_reversed_nonadjacent_wires_keep_complex_phase(self):
        amplitudes = Circuit(3, [H(0), H(2), S(0), CNOT(0, 1)]).run().amplitudes
        observable = Y(2) @ X(0)
        transformed = observable.apply(amplitudes, 3)
        expected = [0j] * len(amplitudes)
        for index, value in enumerate(amplitudes):
            bit = index & 1
            expected[index ^ 0b101] += (1j if bit == 0 else -1j) * value
        self.assertEqual(transformed, expected)
        self.assertEqual(observable.terms[0][1][0][0], (2,))

    def test_nonunitary_hermitian_and_multiqubit_factor(self):
        observable = Hermitian((0,), ((2, 1j), (-1j, 3)))
        self.assertEqual(observable.apply([1 + 0j, 0j], 1), [2 + 0j, -1j])
        with self.assertRaises(ValueError): Gate('not a gate', (0,), ((2, 1j), (-1j, 3)))
        swap = as_observable(SWAP(2, 0))
        amplitudes = [0j] * 8
        amplitudes[1] = 1
        expected = [0j] * 8
        expected[4] = 1
        self.assertEqual(swap.apply(amplitudes, 3), expected)

    def test_observable_snapshots_gate_wires_and_matrix(self):
        gate = Z(0)
        observable = gate + X(0)
        bound = observable.bind()
        gate.wires = (1,)
        gate._matrix = X(0).matrix
        observable.terms = ()
        self.assertEqual(bound.apply([1 + 0j, 0j], 1), [1 + 0j, 1 + 0j])

    def test_matrix_constructor_copies_input(self):
        wires = [0]
        matrix = [[2, 1j], [-1j, 3]]
        observable = Hermitian(wires, matrix)
        wires[0] = 9
        matrix[0][0] = 100
        self.assertEqual(observable.apply([1 + 0j, 0j], 1), [2 + 0j, -1j])

    def test_application_does_not_modify_or_alias_input(self):
        amplitudes = [1 + 0j, 0j]
        identity = 0 * Z(0) + 2
        before = amplitudes[:]
        result = identity.apply(amplitudes, 1)
        self.assertEqual(result, [2 + 0j, 0j])
        self.assertEqual(amplitudes, before)
        self.assertIsNot(result, amplitudes)

    def test_reject_overlap_nonhermitian_or_trainable_factors(self):
        with self.assertRaises(ValueError): Z(0) @ X(0)
        with self.assertRaises(ValueError): (Z(0) + Z(1)) @ X(1)
        for observable in (S(0), RY(0, 0.3)):
            with self.assertRaises(ValueError): as_observable(observable)
            with self.assertRaises(ValueError): observable + Z(0)
        for requires_grad in (True, False):
            observable = RY(0, Tensor(0.0, requires_grad=requires_grad))
            with self.assertRaises(TypeError): as_observable(observable)
        with self.assertRaises(TypeError): Z(0) * (1 + 1j)
        with self.assertRaises(TypeError): Z(0) * Tensor(0.3)

    def test_reject_invalid_local_matrix_and_out_of_range_wire(self):
        for wires, matrix in (((0,), ((1, 1j), (1j, 1))), ((0, 0), ((1,),)), ((-1,), ((1, 0), (0, 1))), ((0,), ((1,),))):
            with self.assertRaises(ValueError): Hermitian(wires, matrix)
        with self.assertRaises(ValueError): as_observable(Z(1)).apply([1 + 0j, 0j], 1)
        with self.assertRaises(TypeError): as_observable([[1, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()
