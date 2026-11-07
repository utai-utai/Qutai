import unittest
from math import cos, sin
from unittest.mock import patch

from qutai import Tensor, quantum as q
from qutai.backends import cpu
from qutai.core.context import set_recording


METHODS = ('adjoint', 'parameter-shift')


def initial_state():
    return q.Circuit(3, [q.RY(0, 0.31), q.H(1), q.S(1), q.RX(2, -0.47), q.CNOT(2, 0)]).run()


def rich_circuit(angles):
    """复相位、反序连线、非自逆门与共享角度都影响测量概率。"""
    phase = q.Gate('phase', (2, 0), ((1, 0, 0, 0), (0, 1j, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1j)))
    return q.Circuit(3, [
        q.H(2), q.RX(0, angles[0]), q.CNOT(2, 0), q.RY(1, angles[1]),
        phase, q.RZ(2, angles[2]), q.SWAP(1, 0), q.S(1),
        q.RY(2, angles[0]), q.H(0), q.RZ(0, angles[1]),
    ])


def reference_probabilities(values, wires=None):
    """直接按基态位串分组振幅，独立核对边缘化与比特顺序。"""
    amplitudes = rich_circuit(values).run(initial_state()).amplitudes
    selected = tuple(range(3)) if wires is None else (wires,) if isinstance(wires, int) else tuple(wires)
    probabilities = [0.0] * (2 ** len(selected))
    for index, amplitude in enumerate(amplitudes):
        bits = f'{index:03b}'
        outcome = int(''.join(bits[wire] for wire in selected), 2)
        probabilities[outcome] += abs(amplitude) ** 2
    return probabilities


def weighted_reference(values, wires, upstream):
    return sum(value * weight for value, weight in zip(reference_probabilities(values, wires), upstream, strict=True))


def expression_circuit(weight, bias):
    angle = 2 * weight + bias
    return q.Circuit(2, [q.RY(0, angle), q.CNOT(0, 1), q.RX(1, weight * weight - bias), q.RZ(0, angle)])


def mixed_numeric_objective(weight, bias):
    circuit = expression_circuit(weight, bias)
    probabilities = circuit.probabilities((1, 0))
    return sum(value * value for value in probabilities) + 0.3 * circuit.expectation(q.Z(0))


class QuantumProbabilityTests(unittest.TestCase):
    def assertValues(self, actual, expected, places=11):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            self.assertAlmostEqual(left, right, places=places)

    def test_full_and_reordered_marginal_probabilities_match_basis_reference(self):
        values = [0.23, -0.47, 0.61]
        for wires in (None, 1, (2, 0), (0, 2), (2, 1, 0)):
            expected = reference_probabilities(values, wires)
            for method in METHODS:
                with self.subTest(wires=wires, method=method):
                    result = rich_circuit(values).probabilities(wires, state=initial_state(), method=method)
                    self.assertIsInstance(result, list)
                    self.assertValues(result, expected)
                    self.assertAlmostEqual(sum(result), 1, places=12)
                    self.assertTrue(all(0 <= value <= 1 for value in result))
        self.assertNotEqual(reference_probabilities(values, (2, 0)), reference_probabilities(values, (0, 2)))

    def test_tensor_shape_and_state_numeric_api_are_distinct(self):
        angle = Tensor(0.37, requires_grad=True)
        circuit = q.Circuit(2, [q.RY(0, angle), q.CNOT(0, 1)])
        full, marginal = circuit.probabilities(), circuit.probabilities(1)
        self.assertIsInstance(full, Tensor)
        self.assertEqual(full.shape, (4,))
        self.assertEqual(marginal.shape, (2,))
        self.assertValues(full.values, [cos(0.37 / 2) ** 2, 0, 0, sin(0.37 / 2) ** 2])
        self.assertValues(marginal.values, [cos(0.37 / 2) ** 2, sin(0.37 / 2) ** 2])
        self.assertIsInstance(circuit.run().probabilities(), list)
        self.assertEqual(circuit.parameters(), (angle,))

    def test_ry_probability_jacobian_matches_analytic_derivatives(self):
        for method in METHODS:
            for value in (-0.73, 0.0, 0.49):
                angle = Tensor(value, requires_grad=True)
                probabilities = q.Circuit(1, [q.RY(0, angle)]).probabilities(method=method)
                self.assertValues(probabilities.values, [cos(value / 2) ** 2, sin(value / 2) ** 2])
                probabilities.backward([1, 0])
                self.assertAlmostEqual(angle.grad.item(), -sin(value) / 2, places=12)
                angle.zero_grad()
                probabilities.backward([0, 1])
                self.assertAlmostEqual(angle.grad.item(), sin(value) / 2, places=12)

    def test_weighted_probability_gradients_match_both_methods_and_finite_difference(self):
        values, epsilon = [0.23, -0.47, 0.61], 1e-6
        cases = [(None, [0.2, -0.7, 1.1, 0.4, -0.3, 0.8, -0.6, 0.5]),
                 ((2, 0), [0.37, -0.82, 1.13, -0.2]), (1, [0.4, -0.9])]
        for wires, upstream in cases:
            gradients = []
            for method in METHODS:
                angles = [Tensor(value, requires_grad=True) for value in values]
                result = rich_circuit(angles).probabilities(wires, state=initial_state(), method=method)
                self.assertValues(result.values, reference_probabilities(values, wires))
                result.backward(upstream)
                gradients.append([angle.grad.item() for angle in angles])
            self.assertValues(*gradients)
            for index, derivative in enumerate(gradients[0]):
                high, low = values[:], values[:]
                high[index] += epsilon
                low[index] -= epsilon
                numeric = (weighted_reference(high, wires, upstream) - weighted_reference(low, wires, upstream)) / (2 * epsilon)
                self.assertAlmostEqual(derivative, numeric, places=8)
        self.assertTrue(any(abs(value) > 0.01 for value in gradients[0]))

    def test_probability_sum_has_zero_gradient_for_full_and_partial_outputs(self):
        for method in METHODS:
            for wires in (None, 0, (2, 0)):
                angles = [Tensor(value, requires_grad=True) for value in (0.23, -0.47, 0.61)]
                total = rich_circuit(angles).probabilities(wires, state=initial_state(), method=method).sum()
                self.assertAlmostEqual(total.item(), 1, places=12)
                total.backward()
                self.assertValues([angle.grad.item() for angle in angles], [0, 0, 0])

    def test_shared_angle_occurrences_accumulate_with_original_gradient_shape(self):
        for method in METHODS:
            angle = Tensor([[0.37]], requires_grad=True)
            gate = q.RY(0, angle)
            result = q.Circuit(1, [gate, gate]).probabilities(method=method)
            expected = (-0.3 - 0.6) * sin(0.74)
            result.backward([0.3, -0.6])
            self.assertEqual(angle.grad.shape, (1, 1))
            self.assertAlmostEqual(angle.grad.item(), expected, places=12)
            result.backward([0.3, -0.6])
            self.assertAlmostEqual(angle.grad.item(), 2 * expected, places=12)

    def test_classical_expression_receives_probability_and_expectation_gradients(self):
        derivatives = []
        for method in METHODS:
            weight, bias = Tensor(0.29, requires_grad=True), Tensor(-0.17, requires_grad=True)
            circuit = expression_circuit(weight, bias)
            loss = circuit.probabilities((1, 0), method=method).square().sum() + 0.3 * circuit.expectation(q.Z(0), method=method)
            loss.backward()
            derivatives.append([weight.grad.item(), bias.grad.item()])
        self.assertValues(*derivatives)
        epsilon = 1e-6
        for index, derivative in enumerate(derivatives[0]):
            upper, lower = [0.29, -0.17], [0.29, -0.17]
            upper[index] += epsilon
            lower[index] -= epsilon
            numeric = (mixed_numeric_objective(*upper) - mixed_numeric_objective(*lower)) / (2 * epsilon)
            self.assertAlmostEqual(derivative, numeric, places=8)

    def test_initial_state_and_probability_results_do_not_alias_or_mutate(self):
        state = initial_state()
        before = state.amplitudes[:]
        circuit = rich_circuit([0.23, -0.47, 0.61])
        first = circuit.probabilities((2, 0), state=state)
        first[0] = 100
        self.assertEqual(state.amplitudes, before)
        self.assertValues(circuit.probabilities((2, 0), state=state), reference_probabilities([0.23, -0.47, 0.61], (2, 0)))

    def test_backward_keeps_forward_snapshots_after_every_source_is_changed(self):
        values, upstream, epsilon = [0.23, -0.47, 0.61], [0.37, -0.82, 1.13, -0.2], 1e-6
        high, low = values[:], values[:]
        high[0] += epsilon
        low[0] -= epsilon
        expected = (weighted_reference(high, (2, 0), upstream) - weighted_reference(low, (2, 0), upstream)) / (2 * epsilon)
        for method in METHODS:
            angle = Tensor(values[0], requires_grad=True)
            circuit, state, wires = rich_circuit([angle, values[1], values[2]]), initial_state(), [2, 0]
            result = circuit.probabilities(wires, state=state, method=method)
            angle.values[0] = 1.7
            circuit.gates[1].wires = (2,)
            circuit.gates[4].wires = (0, 1)
            circuit.gates[:] = [q.X(0)]
            state.apply(q.X(1))
            wires[:] = [0, 2]
            result.backward(upstream)
            self.assertAlmostEqual(angle.grad.item(), expected, places=8)
            self.assertEqual(angle.item(), 1.7)

    def test_multiple_forward_graphs_and_repeated_backward_are_independent(self):
        for method in METHODS:
            angle = Tensor(0.3, requires_grad=True)
            circuit = q.Circuit(1, [q.RY(0, angle), q.S(0)])
            first = circuit.probabilities(method=method)
            angle.values[0] = -0.8
            second = circuit.probabilities(method=method)
            (first[1] + second[1]).backward()
            self.assertAlmostEqual(angle.grad.item(), (sin(0.3) + sin(-0.8)) / 2, places=12)
            angle.zero_grad()
            first[1].backward()
            first[1].backward()
            self.assertAlmostEqual(angle.grad.item(), sin(0.3), places=12)

    def test_recording_and_frozen_angles_preserve_numeric_tensor_contract(self):
        for method in METHODS:
            frozen = Tensor(0.37)
            result = q.Circuit(1, [q.RY(0, frozen)]).probabilities(method=method)
            self.assertIsInstance(result, Tensor)
            self.assertFalse(result.requires_grad)
            self.assertEqual(result.parents, ())
            angle = Tensor(0.23, requires_grad=True)
            circuit = q.Circuit(2, [q.RY(0, angle), q.RX(1, frozen), q.CNOT(0, 1)])
            with set_recording(False):
                detached = circuit.probabilities(0, method=method)
            self.assertFalse(detached.requires_grad)
            self.assertEqual(detached.parents, ())
            self.assertTrue(angle.requires_grad)
            circuit.probabilities(0, method=method).backward([0.3, -0.6])
            self.assertIsNotNone(angle.grad)
            self.assertIsNone(frozen.grad)

    def test_probability_backward_requires_matching_vector_seed(self):
        angle = Tensor(0.37, requires_grad=True)
        result = q.Circuit(2, [q.RY(0, angle)]).probabilities()
        for gradient in (None, [1], [[1, 1], [1, 1]]):
            with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                result.backward(gradient)
        result.backward([0.3, -0.2, 0.7, 0.4])
        self.assertIsNotNone(angle.grad)

    def test_probability_readout_does_not_construct_dense_projectors(self):
        for method in METHODS:
            angle = Tensor(0.37, requires_grad=True)
            circuit = q.Circuit(3, [q.RY(0, angle), q.CNOT(0, 2), q.H(1)])
            with patch.object(q.Observable, 'apply', side_effect=AssertionError('概率读出不应构造观测量投影矩阵')):
                result = circuit.probabilities((2, 0), method=method)
                result.backward([0.3, -0.2, 0.7, 0.4])
            self.assertIsNotNone(angle.grad)

    def test_evolution_work_is_shared_by_all_probability_outputs(self):
        counts = {}
        for method in METHODS:
            for wires in (None, (2, 0)):
                angles = [Tensor(value, requires_grad=True) for value in (0.23, -0.47, 0.61)]
                circuit = rich_circuit(angles)
                length, rotations = len(circuit.gates), 5
                with patch.object(cpu, 'apply_gate', wraps=cpu.apply_gate) as apply:
                    result = circuit.probabilities(wires, state=initial_state(), method=method)
                    forward = apply.call_count
                    result.backward([0.1 * index for index in range(result.size)])
                    counts[method, wires] = apply.call_count - forward
                if method == 'adjoint':
                    self.assertLessEqual(counts[method, wires], 2 * length + rotations)
                else:
                    self.assertEqual(counts[method, wires], 2 * rotations * length)
            self.assertEqual(counts[method, None], counts[method, (2, 0)])

    def test_invalid_wire_selection_initial_state_and_method_have_chinese_errors(self):
        for angle in (0.3, Tensor(0.3, requires_grad=True)):
            circuit = q.Circuit(2, [q.RY(0, angle)])
            for wires in ((), (0, 0), -1, 2, (1, 2)):
                with self.subTest(wires=wires), self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    circuit.probabilities(wires)
            for method in ('', 'unknown', 'backprop'):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    circuit.probabilities(method=method)
            with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                circuit.probabilities(state=q.State(1))


if __name__ == '__main__': unittest.main()
