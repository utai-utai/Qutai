"""伴随求导、参数移位与独立有限差分，以及线性的电路遍历次数。"""

import unittest
from math import cos, sin
from unittest.mock import patch

from qutai import Tensor, nn, qnn, quantum as q, stack
from qutai.backends import cpu
from qutai.core.context import set_recording


METHODS = ('adjoint', 'parameter-shift')


def initial_state():
    """包含复相位的非零初态，避免只在计算基初态验证。"""
    return q.Circuit(3, [q.RY(0, 0.31), q.H(1), q.S(1), q.RX(2, -0.47), q.CNOT(2, 0)]).run()


def rich_circuit(angles):
    """所有旋转轴、反序连线与非自逆的固定局部门。"""
    phase = q.Gate('phase', (2, 0), ((1, 0, 0, 0), (0, 1j, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1j)))
    return q.Circuit(3, [
        q.H(2), q.RX(0, angles[0]), q.CNOT(2, 0), q.RY(1, angles[1]),
        phase, q.RZ(2, angles[2]), q.SWAP(1, 0), q.S(1),
        q.RY(2, angles[0]), q.H(0), q.T(2), q.RZ(0, angles[1]),
    ])


def observables():
    return [
        q.Z(1),
        q.Hermitian((2, 0), ((0.4, 0.2j, 0, 0.3), (-0.2j, -0.7, 0.1, 0),
                              (0, 0.1, 0.2, 0.4j), (0.3, 0, -0.4j, 0.6))),
        0.3 * q.Z(0) - 0.8 * (q.Y(0) @ q.X(2)) + 0.2,
    ]


def numeric_objective(values, weights):
    result = rich_circuit(values).expectation(observables(), state=initial_state())
    return sum(value * weight for value, weight in zip(result, weights, strict=True))


def expression_circuit(weight, bias):
    angle = 2 * weight + bias
    return q.Circuit(2, [q.RY(0, angle), q.CNOT(0, 1), q.RX(1, weight * weight - bias), q.RZ(0, angle)])


def counted_circuit(depth):
    gates = [q.H(0), q.H(1)]
    for index in range(depth):
        gates.extend([q.RX(0, Tensor(0.21 + index * 0.03, requires_grad=True)),
                      q.RY(1, Tensor(-0.32 + index * 0.02, requires_grad=True)), q.CNOT(0, 1),
                      q.RZ(0, Tensor(0.17 - index * 0.01, requires_grad=True))])
    return q.Circuit(2, gates)


class AdjointTests(unittest.TestCase):
    def assertValues(self, actual, expected, places=11):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            self.assertAlmostEqual(left, right, places=places)

    def test_gate_adjoint_inverts_complex_nonselfinverse_gates(self):
        state = initial_state()
        cases = [q.S(2), q.T(0), q.RX(1, 0.43), q.RY(0, -0.29), q.RZ(2, 0.71),
                 q.Gate('phase', (2, 0), ((1, 0, 0, 0), (0, 1j, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1j)))]
        for gate in cases:
            with self.subTest(gate=gate.name):
                after = cpu.apply_gate(state.amplitudes, 3, gate.wires, gate.matrix)
                restored = cpu.apply_gate(after, 3, gate.wires, gate.adjoint_matrix)
                self.assertValues([abs(value - target) for value, target in zip(restored, state.amplitudes)], [0] * 8)
                self.assertEqual(gate.adjoint_matrix, tuple(tuple(value.conjugate() for value in row) for row in zip(*gate.matrix)))

    def test_rotation_derivative_matrices_match_entrywise_finite_differences(self):
        epsilon = 1e-6
        for factory in (q.RX, q.RY, q.RZ):
            for angle in (-0.73, 0.0, 0.49):
                derivative = factory(0, angle).derivative_matrix
                upper, lower = factory(0, angle + epsilon).matrix, factory(0, angle - epsilon).matrix
                for row in range(2):
                    for column in range(2):
                        with self.subTest(axis=factory.__name__, angle=angle, row=row, column=column):
                            self.assertAlmostEqual(abs(derivative[row][column] - (upper[row][column] - lower[row][column]) / (2 * epsilon)), 0, places=9)

    def test_weighted_vector_adjoint_matches_shift_and_independent_finite_difference(self):
        values, upstream, epsilon = [0.23, -0.47, 0.61], [0.37, -0.82, 1.13], 1e-6
        gradients, predictions = [], []
        for method in METHODS:
            angles = [Tensor(value, requires_grad=True) for value in values]
            result = rich_circuit(angles).expectation(observables(), state=initial_state(), method=method)
            self.assertEqual(result.shape, (3,))
            result.backward(upstream)
            predictions.append(result.values)
            gradients.append([angle.grad.item() for angle in angles])
        self.assertValues(predictions[0], predictions[1])
        self.assertValues(gradients[0], gradients[1])
        for index, derivative in enumerate(gradients[0]):
            high, low = values[:], values[:]
            high[index] += epsilon
            low[index] -= epsilon
            numeric = (numeric_objective(high, upstream) - numeric_objective(low, upstream)) / (2 * epsilon)
            self.assertAlmostEqual(derivative, numeric, places=8)
        self.assertTrue(any(abs(value) > 0.1 for value in gradients[0]))

    def test_shared_gate_occurrences_accumulate_and_preserve_parent_shape(self):
        for method in METHODS:
            angle = Tensor([[0.37]], requires_grad=True)
            gate = q.RY(0, angle)
            result = q.Circuit(1, [gate, gate]).expectation([q.Z(0), q.X(0)], method=method)
            expected = -0.6 * sin(0.74) - 1.2 * cos(0.74)
            result.backward([0.3, -0.6])
            self.assertEqual(angle.grad.shape, (1, 1))
            self.assertAlmostEqual(angle.grad.item(), expected, places=12)
            result.backward([0.3, -0.6])
            self.assertAlmostEqual(angle.grad.item(), 2 * expected, places=12)

    def test_classical_expressions_receive_shared_angle_gradients(self):
        observable = 0.4 * q.Z(0) + 0.7 * (q.X(0) @ q.X(1)) - 0.2 * q.Y(1)
        derivatives = []
        for method in METHODS:
            weight, bias = Tensor(0.29, requires_grad=True), Tensor(-0.17, requires_grad=True)
            loss = expression_circuit(weight, bias).expectation(observable, method=method).square()
            loss.backward()
            derivatives.append([weight.grad.item(), bias.grad.item()])
        self.assertValues(*derivatives)
        epsilon = 1e-6
        for index, derivative in enumerate(derivatives[0]):
            upper, lower = [0.29, -0.17], [0.29, -0.17]
            upper[index] += epsilon
            lower[index] -= epsilon
            numeric = (expression_circuit(*upper).expectation(observable) ** 2
                       - expression_circuit(*lower).expectation(observable) ** 2) / (2 * epsilon)
            self.assertAlmostEqual(derivative, numeric, places=8)

    def test_backward_preserves_all_forward_snapshots_after_source_mutation(self):
        weights = [0.37, -0.82, 1.13]
        epsilon = 1e-6
        expected = (numeric_objective([0.23 + epsilon, -0.47, 0.61], weights)
                    - numeric_objective([0.23 - epsilon, -0.47, 0.61], weights)) / (2 * epsilon)
        for method in METHODS:
            angle = Tensor(0.23, requires_grad=True)
            circuit, state, readouts = rich_circuit([angle, -0.47, 0.61]), initial_state(), observables()
            result = circuit.expectation(readouts, state=state, method=method)
            circuit.gates[1].wires = (2,)
            circuit.gates[2].wires = (1, 0)
            circuit.gates[4].wires = (0, 1)
            circuit.gates[:] = [q.X(0)]
            angle.values[0] = 1.7
            state.apply(q.X(1))
            readouts[0].wires = (2,)
            readouts[1].terms = ()
            readouts.clear()
            result.backward(weights)
            self.assertAlmostEqual(angle.grad.item(), expected, places=8)
            self.assertEqual(angle.item(), 1.7)

    def test_multiple_forward_graphs_and_repeated_backward_remain_independent(self):
        for method in METHODS:
            angle = Tensor(0.3, requires_grad=True)
            circuit = q.Circuit(1, [q.RY(0, angle), q.S(0)])
            first = circuit.expectation(q.Y(0), method=method)
            angle.values[0] = -0.8
            second = circuit.expectation(q.Y(0), method=method)
            (first + second).backward()
            self.assertAlmostEqual(angle.grad.item(), cos(0.3) + cos(-0.8), places=12)
            angle.zero_grad()
            first.backward()
            first.backward()
            self.assertAlmostEqual(angle.grad.item(), 2 * cos(0.3), places=12)

    def test_custom_gate_named_rotation_uses_actual_inverse_without_derivative(self):
        fixed = q.Gate('RY', (0,), q.S(0).matrix, angle=0.7)
        for method in METHODS:
            angle = Tensor(0.37, requires_grad=True)
            circuit = q.Circuit(1, [q.RY(0, angle), fixed])
            result = circuit.expectation(q.Y(0), method=method)
            self.assertAlmostEqual(result.item(), sin(0.37), places=12)
            result.backward()
            self.assertAlmostEqual(angle.grad.item(), cos(0.37), places=12)
            self.assertEqual(circuit.parameters(), (angle,))
        with self.assertRaises(TypeError): _ = fixed.derivative_matrix

    def test_frozen_angles_and_recording_context_keep_original_semantics(self):
        for method in METHODS:
            first, frozen = Tensor(0.23, requires_grad=True), Tensor(-0.47)
            circuit = rich_circuit([first, frozen, 0.61])
            with set_recording(False):
                detached = circuit.expectation(observables(), method=method)
            self.assertFalse(detached.requires_grad)
            self.assertEqual(detached.parents, ())
            self.assertTrue(first.requires_grad)
            result = circuit.expectation(observables(), method=method)
            result.backward([0.37, -0.82, 1.13])
            self.assertIsNotNone(first.grad)
            self.assertIsNone(frozen.grad)
            self.assertFalse(frozen.requires_grad)

    def test_adjoint_gate_application_count_grows_linearly(self):
        counts = {}
        for depth in (2, 4):
            for method in METHODS:
                circuit = counted_circuit(depth)
                gate_count, rotation_count = len(circuit.gates), 3 * depth
                with patch.object(cpu, 'apply_gate', wraps=cpu.apply_gate) as apply:
                    result = circuit.expectation(q.Z(0), method=method)
                    result.backward()
                counts[depth, method] = apply.call_count
                if method == 'adjoint':
                    # 前向 L；读出和伴随初始化各一次；逆序两个态及 R 个局部导数。
                    self.assertLessEqual(apply.call_count, 3 * gate_count + rotation_count + 2)
                else:
                    self.assertEqual(apply.call_count, (1 + 2 * rotation_count) * (gate_count + 1))
        self.assertLess(counts[4, 'adjoint'], 2 * counts[2, 'adjoint'])
        self.assertGreater(counts[4, 'parameter-shift'], 3 * counts[2, 'parameter-shift'])
        circuit = counted_circuit(2)
        with patch.object(cpu, 'apply_gate', wraps=cpu.apply_gate) as apply:
            circuit.expectation(q.Z(0)).backward()
        self.assertEqual(apply.call_count, counts[2, 'adjoint'])

    def test_qaoa_shared_gamma_and_beta_match_parameter_shift(self):
        algorithm = q.QAOA(3, 0.3 * q.Z(0) + q.maxcut([(0, 1), (1, 2), (2, 0)]), depth=2, seed=7)
        gradients = []
        for method in METHODS:
            algorithm.weight.zero_grad()
            algorithm.circuit().expectation(algorithm.cost, method=method).backward()
            gradients.append(algorithm.weight.grad.values[:])
        self.assertValues(*gradients)
        self.assertTrue(any(abs(value) > 0.1 for value in gradients[0]))
        algorithm.weight.zero_grad()
        algorithm.expectation().backward()
        self.assertValues(algorithm.weight.grad.values, gradients[0])

    def test_hybrid_network_inputs_and_all_parameters_match_parameter_shift(self):
        first, pqc, last = nn.Linear(2, 2, seed=7), qnn.PQC(2, depth=2, seed=42), nn.Linear(2, 1, seed=11)
        model = nn.Model([first, pqc, last])
        inputs = Tensor([[0.2, -0.3], [0.4, 0.1]], requires_grad=True)
        model(inputs).square().mean().backward()
        parameters = (inputs, *model.parameters())
        expected = [parameter.grad.values[:] for parameter in parameters]
        for parameter in parameters: parameter.zero_grad()
        encoded = first(inputs)
        features = stack([pqc.circuit(row).expectation([q.Z(0), q.Z(1)], method='parameter-shift') for row in encoded])
        last(features).square().mean().backward()
        for parameter, values in zip(parameters, expected, strict=True):
            self.assertValues(parameter.grad.values, values)

    def test_both_methods_preserve_scalar_vector_numeric_and_frozen_returns(self):
        for method in METHODS:
            numeric = q.Circuit(1, [q.RY(0, 0.37)])
            self.assertIsInstance(numeric.expectation(q.Z(0), method=method), float)
            self.assertIsInstance(numeric.expectation([q.Z(0)], method=method), list)
            frozen = q.Circuit(1, [q.RY(0, Tensor(0.37))])
            scalar = frozen.expectation(q.Z(0), method=method)
            vector = frozen.expectation([q.Z(0)], method=method)
            self.assertEqual(scalar.shape, ())
            self.assertEqual(vector.shape, (1,))
            self.assertFalse(scalar.requires_grad)
            self.assertFalse(vector.requires_grad)

    def test_invalid_gradient_method_has_a_chinese_error(self):
        for angle in (0.3, Tensor(0.3, requires_grad=True)):
            circuit = q.Circuit(1, [q.RY(0, angle)])
            for invalid in ('', 'unknown', 'backprop'):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    circuit.expectation(q.Z(0), method=invalid)


if __name__ == '__main__': unittest.main()
