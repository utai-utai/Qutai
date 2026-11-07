import unittest
from math import cos, sin, sqrt

from qutai import Tensor, nn, quantum as q
from qutai.core.context import set_recording


def interacting_circuit(angles):
    """包含复相位、非相邻和反序控制位的三比特电路。"""
    return q.Circuit(3, [
        q.H(2), q.RX(0, angles[0]), q.CNOT(0, 2),
        q.RY(1, angles[1]), q.CNOT(2, 1), q.RZ(2, angles[2]),
        q.H(0), q.SWAP(0, 1),
    ])


class QuantumExpectationTests(unittest.TestCase):
    def test_state_expectations_preserve_complex_state(self):
        cases = [
            (q.State(1), q.Z(0), 1),
            (q.State(1).apply(q.X(0)), q.Z(0), -1),
            (q.State(1, [1 / sqrt(2), 1 / sqrt(2)]), q.X(0), 1),
            (q.State(1, [1 / sqrt(2), 1j / sqrt(2)]), q.Y(0), 1),
            (q.State(1, [1 / sqrt(2), -1j / sqrt(2)]), q.Y(0), -1),
        ]
        for state, observable, expected in cases:
            with self.subTest(observable=observable.name, expected=expected):
                before = state.amplitudes[:]
                result = state.expectation(observable)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected, places=12)
                self.assertEqual(state.amplitudes, before)

    def test_exact_expectation_correlations_and_custom_hermitian_gate(self):
        bell = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run()
        self.assertAlmostEqual(bell.expectation(q.Z(0)), 0, places=12)
        self.assertAlmostEqual(bell.expectation(q.Z(1)), 0, places=12)
        zz = q.Gate('ZZ', (1, 0), ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
        self.assertAlmostEqual(bell.expectation(zz), 1, places=12)
        self.assertAlmostEqual(bell.expectation(q.SWAP(1, 0)), 1, places=12)

    def test_float_and_tensor_angle_return_types_and_recording_context(self):
        self.assertIsInstance(q.Circuit(1, [q.RY(0, 0.3)]).expectation(q.Z(0)), float)
        frozen = Tensor(0.3)
        circuit = q.Circuit(1, [q.RY(0, frozen)])
        result = circuit.expectation(q.Z(0))
        self.assertIsInstance(result, Tensor)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())
        self.assertEqual(circuit.parameters(), (frozen,))

        angle = Tensor(0.3, requires_grad=True)
        circuit = q.Circuit(1, [q.RY(0, angle)])
        with set_recording(False):
            detached = circuit.expectation(q.Z(0))
        self.assertIsInstance(detached, Tensor)
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.parents, ())
        self.assertTrue(angle.requires_grad)
        self.assertTrue(circuit.expectation(q.Z(0)).requires_grad)

    def test_reject_nonhermitian_or_trainable_observables(self):
        state = q.State(1)
        circuit = q.Circuit(1)
        for observable in (q.S(0), q.T(0), q.RX(0, 0.3)):
            with self.subTest(observable=observable.name), self.assertRaises(ValueError):
                state.expectation(observable)
            with self.assertRaises(ValueError): circuit.expectation(observable)
        for requires_grad in (False, True):
            observable = q.RY(0, Tensor(0.0, requires_grad=requires_grad))
            with self.assertRaises(TypeError): state.expectation(observable)
            with self.assertRaises(TypeError): circuit.expectation(observable)
        with self.assertRaises(ValueError): state.expectation(q.Z(1))


class QuantumGradientTests(unittest.TestCase):
    def test_ry_z_matches_cosine_and_analytic_derivative(self):
        angle = Tensor(0.37, requires_grad=True)
        result = q.Circuit(1, [q.RY(0, angle)]).expectation(q.Z(0))
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result.item(), cos(0.37), places=12)
        result.backward()
        self.assertAlmostEqual(angle.grad.item(), -sin(0.37), places=12)

    def test_rx_and_rz_complex_phase_gradients(self):
        cases = [
            ([q.RX], q.Y(0), -sin(0.4), -cos(0.4)),
            ([q.RX], q.Z(0), cos(0.4), -sin(0.4)),
            ([q.H(0), q.RZ], q.X(0), cos(0.4), -sin(0.4)),
            ([q.H(0), q.RZ], q.Y(0), sin(0.4), cos(0.4)),
        ]
        for template, observable, expected, derivative in cases:
            with self.subTest(observable=observable.name, template=str(template)):
                angle = Tensor(0.4, requires_grad=True)
                gates = [gate(0, angle) if callable(gate) else gate for gate in template]
                result = q.Circuit(1, gates).expectation(observable)
                self.assertAlmostEqual(result.item(), expected, places=12)
                result.backward()
                self.assertAlmostEqual(angle.grad.item(), derivative, places=12)

    def test_scalar_angle_in_single_item_tensor_keeps_gradient_shape(self):
        angle = Tensor([[0.4]], requires_grad=True)
        q.Circuit(1, [q.RY(0, angle)]).expectation(q.Z(0)).backward()
        self.assertEqual(angle.grad.shape, (1, 1))
        self.assertAlmostEqual(angle.grad.item(), -sin(0.4), places=12)
        with self.assertRaises(ValueError): q.RY(0, Tensor([0.1, 0.2]))

    def test_live_angle_and_bound_gate_snapshot(self):
        angle = Tensor(0.3, requires_grad=True)
        gate = q.RY(0, angle)
        snapshot = gate.bind()
        initial_matrix = snapshot.matrix
        self.assertIs(gate.angle, angle)
        self.assertIsInstance(snapshot.angle, float)
        angle.values[0] = 0.9
        self.assertNotEqual(gate.matrix, initial_matrix)
        self.assertEqual(snapshot.matrix, initial_matrix)
        self.assertAlmostEqual(q.Circuit(1, [gate]).run().expectation(q.Z(0)), cos(0.9), places=12)
        self.assertAlmostEqual(q.State(1).apply(gate.bind(0.7)).expectation(q.Z(0)), cos(0.7), places=12)
        fixed = q.H(0)
        self.assertIsNot(fixed.bind(), fixed)
        self.assertEqual(fixed.bind().matrix, fixed.matrix)

    def test_shared_angle_and_repeated_gate_accumulate_per_occurrence(self):
        for reuse_gate in (False, True):
            with self.subTest(reuse_gate=reuse_gate):
                angle = Tensor(0.37, requires_grad=True)
                first = q.RY(0, angle)
                circuit = q.Circuit(1, [first, first if reuse_gate else q.RY(0, angle)])
                self.assertEqual(circuit.parameters(), (angle,))
                result = circuit.expectation(q.Z(0))
                self.assertAlmostEqual(result.item(), cos(2 * 0.37), places=12)
                result.backward()
                self.assertAlmostEqual(angle.grad.item(), -2 * sin(2 * 0.37), places=12)

    def test_custom_rotation_name_does_not_change_its_matrix(self):
        gate = q.Gate('RY', (0,), q.H(0).matrix, angle=0.7)
        self.assertEqual(gate.bind().matrix, q.H(0).matrix)
        self.assertAlmostEqual(q.State(1).apply(gate).expectation(q.X(0)), 1.0, places=12)

    def test_quantum_gradient_flows_through_classical_expression(self):
        weight = Tensor(0.2, requires_grad=True)
        bias = Tensor(-0.1, requires_grad=True)
        angle = 2 * weight + bias
        result = q.Circuit(1, [q.RY(0, angle)]).expectation(q.Z(0))
        result.backward()
        self.assertAlmostEqual(weight.grad.item(), -2 * sin(0.3), places=12)
        self.assertAlmostEqual(bias.grad.item(), -sin(0.3), places=12)

    def test_classical_layer_gets_quantum_loss_gradients(self):
        layer = nn.Linear(2, 1, seed=42)
        angle = layer([[0.4, -0.7]])
        result = q.Circuit(1, [q.RY(0, angle)]).expectation(q.Z(0))
        loss = (result - 0.5).square()
        expected = -2 * (cos(angle.item()) - 0.5) * sin(angle.item())
        loss.backward()
        self.assertAlmostEqual(layer.weight.grad.values[0], expected * 0.4, places=12)
        self.assertAlmostEqual(layer.weight.grad.values[1], expected * -0.7, places=12)
        self.assertAlmostEqual(layer.bias.grad.item(), expected, places=12)

    def test_multiqubit_rotations_match_central_differences(self):
        values = [0.23, -0.47, 0.61]
        epsilon = 1e-6
        for wire in range(3):
            for observable_factory in (q.X, q.Y, q.Z):
                observable = observable_factory(wire)
                angles = [Tensor(value, requires_grad=True) for value in values]
                result = interacting_circuit(angles).expectation(observable)
                result.backward()
                for index, angle in enumerate(angles):
                    with self.subTest(wire=wire, observable=observable.name, parameter=index):
                        high, low = values[:], values[:]
                        high[index] += epsilon
                        low[index] -= epsilon
                        numeric = (interacting_circuit(high).expectation(observable)
                                   - interacting_circuit(low).expectation(observable)) / (2 * epsilon)
                        self.assertAlmostEqual(angle.grad.item(), numeric, places=8)

    def test_backward_uses_forward_angles_structure_and_input_state(self):
        angle = Tensor(0.37, requires_grad=True)
        initial = q.State(1, [0, 1])
        circuit = q.Circuit(1, [q.RY(0, angle)])
        result = circuit.expectation(q.Z(0), state=initial)
        self.assertAlmostEqual(result.item(), -cos(0.37), places=12)
        angle.values[0] = 1.2
        circuit.gates[:] = [q.X(0)]
        initial.apply(q.X(0))
        result.backward()
        self.assertAlmostEqual(angle.grad.item(), sin(0.37), places=12)
        self.assertEqual(angle.item(), 1.2)
        self.assertEqual(initial.amplitudes, [1, 0])

    def test_backward_uses_forward_observable_and_numeric_gate_snapshots(self):
        angle = Tensor(0.3, requires_grad=True)
        trainable = q.RY(0, angle)
        fixed = q.RY(1, 0.4)
        observable = q.Z(1)
        circuit = q.Circuit(2, [trainable, q.CNOT(0, 1), fixed])
        result = circuit.expectation(observable)
        self.assertAlmostEqual(result.item(), cos(0.3) * cos(0.4), places=12)
        trainable.wires = (1,)
        fixed.angle = 1.2
        observable.wires = (0,)
        result.backward()
        self.assertAlmostEqual(angle.grad.item(), -sin(0.3) * cos(0.4), places=12)
        self.assertEqual(trainable.wires, (1,))
        self.assertEqual(fixed.angle, 1.2)
        self.assertEqual(observable.wires, (0,))

    def test_separate_forward_graphs_and_repeated_backward_accumulate(self):
        angle = Tensor(0.3, requires_grad=True)
        circuit = q.Circuit(1, [q.RY(0, angle)])
        first = circuit.expectation(q.Z(0))
        angle.values[0] = 0.8
        second = circuit.expectation(q.Z(0))
        (first + second).backward()
        self.assertAlmostEqual(angle.grad.item(), -sin(0.3) - sin(0.8), places=12)
        angle.zero_grad()
        first.backward()
        first.backward()
        self.assertAlmostEqual(angle.grad.item(), -2 * sin(0.3), places=12)

    def test_two_sgd_updates_use_latest_parameter_values(self):
        angle = Tensor(0.3, requires_grad=True)
        circuit = q.Circuit(1, [q.RY(0, angle)])
        self.assertFalse(callable(circuit))
        optimizer = nn.SGD(circuit, learning_rate=0.3)
        loss_function = nn.MSE()
        initial = loss_function(circuit.expectation(q.Z(0)), 0.5).item()
        expected_angle = 0.3
        for _ in range(2):
            optimizer.zero_grad()
            loss = loss_function(circuit.expectation(q.Z(0)), 0.5)
            loss.backward()
            expected_angle += 0.6 * (cos(expected_angle) - 0.5) * sin(expected_angle)
            optimizer.step()
            self.assertAlmostEqual(angle.item(), expected_angle, places=12)
        final = loss_function(circuit.expectation(q.Z(0)), 0.5).item()
        self.assertLess(final, initial)
        self.assertIs(optimizer.parameters[0], angle)



if __name__ == '__main__':
    unittest.main()
