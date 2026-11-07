import unittest
from math import cos, isfinite, sin, sqrt
from unittest.mock import patch

from qutai import Tensor, nn, qnn, quantum
from qutai.core.context import is_recording, set_recording


def sample_circuit(angles):
    """复相位、纠缠和共享角度同时出现的小电路。"""
    return quantum.Circuit(2, [
        quantum.RY(0, 0.63), quantum.RX(1, angles[0]), quantum.CNOT(0, 1),
        quantum.RY(0, angles[1]), quantum.RZ(1, angles[0]), quantum.H(1),
    ])


def sample_observables():
    return [
        quantum.Z(0), quantum.X(1),
        0.4 * quantum.Z(0) - 0.7 * (quantum.X(0) @ quantum.X(1))
        + 0.2 * (quantum.Y(0) @ quantum.Y(1)),
    ]


def weighted_value(angles, weights):
    values = sample_circuit(angles).expectation(sample_observables())
    return sum(value * weight for value, weight in zip(values, weights, strict=True))


class MultiExpectationTests(unittest.TestCase):
    def assertValues(self, actual, expected, places=11):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected, strict=True):
            self.assertAlmostEqual(value, target, places=places)

    def test_numeric_state_and_circuit_readout_shapes_and_values(self):
        state = quantum.State(1, [1 / sqrt(2), 1j / sqrt(2)])
        observables = [quantum.X(0), quantum.Y(0), quantum.Z(0)]
        before = state.amplitudes[:]
        self.assertValues(state.expectation(observables), [0, 1, 0])
        self.assertEqual(state.amplitudes, before)
        self.assertIsInstance(state.expectation(quantum.Y(0)), float)
        self.assertValues(state.expectation([quantum.Y(0)]), [1])

        circuit = sample_circuit([0.23, -0.47])
        observables = sample_observables()
        expected = [circuit.expectation(observable) for observable in observables]
        result = circuit.expectation(observables)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(value, float) for value in result))
        self.assertValues(result, expected)
        self.assertValues(circuit.run().expectation(observables), expected)

    def test_iterators_are_consumed_once_and_keep_observable_order(self):
        circuit = quantum.Circuit(1, [quantum.RY(0, 0.31)])
        expected = [cos(0.31), sin(0.31), 0]
        self.assertValues(circuit.expectation(gate(0) for gate in (quantum.Z, quantum.X, quantum.Y)), expected)
        self.assertValues(circuit.run().expectation(gate(0) for gate in (quantum.Z, quantum.X, quantum.Y)), expected)

    def test_tensor_scalar_and_single_item_vector_remain_distinct(self):
        angle = Tensor(0.37, requires_grad=True)
        circuit = quantum.Circuit(1, [quantum.RY(0, angle)])
        scalar = circuit.expectation(quantum.Z(0))
        vector = circuit.expectation([quantum.Z(0)])
        self.assertIsInstance(scalar, Tensor)
        self.assertIsInstance(vector, Tensor)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(vector.shape, (1,))
        self.assertAlmostEqual(scalar.item(), vector.item(), places=12)
        vector.backward([2.5])
        self.assertAlmostEqual(angle.grad.item(), -2.5 * sin(0.37), places=12)

    def test_weighted_vector_gradient_matches_central_differences(self):
        values, weights, epsilon = [0.23, -0.47], [0.3, -0.8, 1.2], 1e-6
        angles = [Tensor(value, requires_grad=True) for value in values]
        result = sample_circuit(angles).expectation(sample_observables())
        self.assertEqual(result.shape, (3,))
        result.backward(weights)
        for index, angle in enumerate(angles):
            high, low = values[:], values[:]
            high[index] += epsilon
            low[index] -= epsilon
            expected = (weighted_value(high, weights) - weighted_value(low, weights)) / (2 * epsilon)
            self.assertTrue(isfinite(angle.grad.item()))
            self.assertAlmostEqual(angle.grad.item(), expected, places=8)

    def test_repeated_angle_contributes_for_each_gate_and_each_output(self):
        angle = Tensor(0.37, requires_grad=True)
        gate = quantum.RY(0, angle)
        circuit = quantum.Circuit(1, [gate, gate])
        result = circuit.expectation([quantum.Z(0), quantum.X(0)])
        self.assertValues(result.values, [cos(0.74), sin(0.74)])
        result.backward([0.3, -0.6])
        expected = -0.6 * sin(0.74) - 1.2 * cos(0.74)
        self.assertAlmostEqual(angle.grad.item(), expected, places=12)
        result.backward([0.3, -0.6])
        self.assertAlmostEqual(angle.grad.item(), 2 * expected, places=12)

    def test_forward_and_backward_share_each_evolved_state_across_readouts(self):
        first, second = Tensor(0.23, requires_grad=True), Tensor(-0.47, requires_grad=True)
        circuit = sample_circuit([first, second])
        gate_count = len(circuit.gates)
        occurrence_count = 3
        with patch.object(quantum.State, 'apply', autospec=True, side_effect=quantum.State.apply) as apply:
            result = circuit.expectation(sample_observables(), method='parameter-shift')
            self.assertEqual(apply.call_count, gate_count)
            result.backward([0.3, -0.8, 1.2])
            self.assertEqual(apply.call_count, gate_count * (1 + 2 * occurrence_count))

    def test_backward_uses_forward_snapshots_of_every_numeric_input(self):
        value, epsilon = 0.37, 1e-6
        weights = [0.7, -0.2, 1.1]
        angle = Tensor(value, requires_grad=True)
        initial = quantum.State(2).apply(quantum.X(1))
        trainable, fixed, observable = quantum.RY(0, angle), quantum.RY(1, 0.4), quantum.Z(1)
        observables = [observable, quantum.X(0) @ quantum.X(1), 0.5 * observable + quantum.Z(0)]
        circuit = quantum.Circuit(2, [trainable, quantum.CNOT(0, 1), fixed])
        numeric = []
        for shifted in (value + epsilon, value - epsilon):
            values = quantum.Circuit(2, [quantum.RY(0, shifted), quantum.CNOT(0, 1), quantum.RY(1, 0.4)]).expectation(observables, state=initial)
            numeric.append(sum(item * weight for item, weight in zip(values, weights, strict=True)))
        expected = (numeric[0] - numeric[1]) / (2 * epsilon)
        result = circuit.expectation(observables, state=initial)
        angle.values[0] = 1.2
        trainable.wires = (1,)
        fixed.angle = 1.1
        observable.wires = (0,)
        observables.clear()
        circuit.gates[:] = [quantum.X(0)]
        initial.apply(quantum.X(0))
        result.backward(weights)
        self.assertAlmostEqual(angle.grad.item(), expected, places=8)
        self.assertEqual(angle.item(), 1.2)
        self.assertEqual(initial.amplitudes, [0j, 0j, 0j, 1 + 0j])

    def test_frozen_and_paused_recording_keep_vector_tensor_without_graph(self):
        frozen = quantum.Circuit(1, [quantum.RY(0, Tensor(0.37))])
        result = frozen.expectation([quantum.Z(0), quantum.X(0)])
        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (2,))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())

        angle = Tensor(0.37, requires_grad=True)
        circuit = quantum.Circuit(1, [quantum.RY(0, angle)])
        with set_recording(False):
            detached = circuit.expectation([quantum.Z(0), quantum.X(0)])
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.parents, ())
        self.assertTrue(angle.requires_grad)
        self.assertTrue(is_recording())
        self.assertTrue(circuit.expectation([quantum.Z(0), quantum.X(0)]).requires_grad)

    def test_reject_empty_invalid_nonhermitian_or_trainable_observables(self):
        state, circuit = quantum.State(1), quantum.Circuit(1)
        for target in (state, circuit):
            for invalid in ([], iter(())):
                with self.subTest(target=type(target).__name__), self.assertRaises(ValueError):
                    target.expectation(invalid)
            for invalid in ([quantum.Z(0), 1], 1):
                with self.assertRaises(TypeError): target.expectation(invalid)
            with self.assertRaises(ValueError): target.expectation([quantum.Z(0), quantum.S(0)])
            with self.assertRaises(ValueError): target.expectation([quantum.Z(1)])
            with self.assertRaises(TypeError): target.expectation([quantum.RY(0, Tensor(0.0))])

    def test_pqc_reads_all_qubits_once_per_sample(self):
        layer = qnn.PQC(3, depth=2, seed=42)
        inputs = Tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, 0.1]], requires_grad=True)
        with patch.object(quantum.Circuit, 'expectation', autospec=True, side_effect=quantum.Circuit.expectation) as expectation:
            result = layer(inputs)
        self.assertEqual(expectation.call_count, 2)
        self.assertEqual(result.shape, inputs.shape)
        result.square().mean().backward()
        self.assertTrue(all(isfinite(value) for value in inputs.grad.values))
        self.assertTrue(all(isfinite(value) for value in layer.weight.grad.values))

    def test_classical_layer_receives_weighted_vector_quantum_gradients(self):
        layer = nn.Linear(2, 2, seed=7)
        inputs, weights, epsilon = [[0.2, -0.3]], [0.4, -0.7, 1.1], 1e-6
        angles = layer(inputs)[0]
        sample_circuit([angles[0], angles[1]]).expectation(sample_observables()).backward(weights)
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            expected = parameter.grad.values[:]
            for index, analytic in enumerate(expected):
                original = parameter.values[index]
                parameter.values[index] = original + epsilon
                upper = weighted_value(layer(inputs).tolist()[0], weights)
                parameter.values[index] = original - epsilon
                lower = weighted_value(layer(inputs).tolist()[0], weights)
                parameter.values[index] = original
                self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)


if __name__ == '__main__': unittest.main()
