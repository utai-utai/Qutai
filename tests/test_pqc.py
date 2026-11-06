"""随机结构固定的 PQC、批次梯度和经典量子混合网络。"""

import random
import unittest
from math import isfinite, pi

from qutai import Tensor, nn, qnn, quantum
from qutai.core.context import is_recording, set_recording
from qutai.visualization import Network


def reference_output(layer, inputs):
    """直接按定义演化一个样本，独立核对编码、门顺序与最终读出。"""
    state = quantum.State(layer.qubits)
    for wire, value in enumerate(inputs): state.apply(quantum.RY(wire, value))
    for depth, rotations in enumerate(layer.rotations):
        for wire, name in enumerate(rotations):
            state.apply(getattr(quantum, name)(wire, layer.weight.values[depth * layer.qubits + wire]))
        for wire in range(layer.qubits - 1):
            state.apply(quantum.CNOT(wire, wire + 1) if layer.entangler == 'cnot' else quantum.CZ(wire, wire + 1))
    for wire in range(layer.qubits): state.apply(quantum.RY(wire, pi / 4))
    return [state.expectation(quantum.Z(wire)) for wire in range(layer.qubits)]


def objective(layer, inputs):
    return layer(inputs, train=False).square().mean().item()


def hybrid_model(entangler='cnot'):
    return nn.Model([nn.Linear(2, 2, seed=7), qnn.PQC(2, depth=2, entangler=entangler, seed=42), nn.Linear(2, 1, seed=11)])


class PQCTests(unittest.TestCase):
    def assertValues(self, actual, expected, places=11):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected, strict=True): self.assertAlmostEqual(value, target, places=places)

    def test_seed_reproduces_axes_and_weights_without_changing_global_random(self):
        previous = random.getstate()
        first, second = qnn.PQC(3, depth=4, seed=42), qnn.PQC(3, depth=4, seed=42)
        self.assertEqual(random.getstate(), previous)
        self.assertEqual(first.rotations, second.rotations)
        self.assertEqual(first.weight.values, second.weight.values)
        self.assertEqual(first.weight.shape, (4, 3))
        self.assertTrue(first.weight.requires_grad)
        self.assertEqual({name for row in first.rotations for name in row}, {'RX', 'RY', 'RZ'})
        structure, weights = first.rotations, first.weight.values[:]
        first([[0.2, -0.4, 0.7], [0.8, 0.1, -0.3]])
        first([0.2, -0.4, 0.7])
        self.assertEqual(first.rotations, structure)
        self.assertEqual(first.weight.values, weights)
        self.assertEqual(random.getstate(), previous)

    def test_circuit_and_features_match_direct_state_evolution(self):
        for entangler in ('cnot', 'cz'):
            for qubits, inputs in ((1, [0.37]), (3, [0.2, -0.4, 0.7])):
                with self.subTest(entangler=entangler, qubits=qubits):
                    layer = qnn.PQC(qubits, depth=2, entangler=entangler, seed=42)
                    expected = reference_output(layer, inputs)
                    result = layer(inputs)
                    self.assertIsInstance(result, Tensor)
                    self.assertEqual(result.shape, (qubits,))
                    self.assertValues(result.values, expected)
                    state = layer.circuit(inputs).run()
                    self.assertValues([state.expectation(quantum.Z(wire)) for wire in range(qubits)], expected)

    def test_cnot_and_cz_are_effective_distinct_configurations(self):
        cnot = qnn.PQC(2, depth=1, entangler='cnot', seed=42)
        cz = qnn.PQC(2, depth=1, entangler='cz', seed=42)
        self.assertEqual(cnot.rotations, cz.rotations)
        self.assertEqual(cnot.weight.values, cz.weight.values)
        self.assertTrue(any(abs(a - b) > 1e-5 for a, b in zip(cnot([0.3, -0.6]).values, cz([0.3, -0.6]).values, strict=True)))

    def test_vector_batch_and_leading_dimensions_keep_shape(self):
        layer = qnn.PQC(2, depth=2, seed=42)
        data = Tensor([0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8], shape=(2, 2, 2))
        result = layer(data)
        self.assertEqual(result.shape, data.shape)
        expected = [value for sample in data.reshape((4, 2)).tolist() for value in reference_output(layer, sample)]
        self.assertValues(result.values, expected)
        self.assertEqual(layer.parameters(), (layer.weight,))
        self.assertNotIn(data, layer.parameters())

    def test_all_weight_and_input_gradients_match_central_differences(self):
        epsilon = 1e-6
        for entangler in ('cnot', 'cz'):
            with self.subTest(entangler=entangler):
                layer = qnn.PQC(3, depth=3, entangler=entangler, seed=42)
                inputs = Tensor([[0.23, -0.47, 0.61], [-0.31, 0.52, 0.17]], requires_grad=True)
                layer(inputs).square().mean().backward()
                for parent in (inputs, layer.weight):
                    expected = parent.grad.values[:]
                    for position, analytic in enumerate(expected):
                        original = parent.values[position]
                        parent.values[position] = original + epsilon
                        upper = objective(layer, inputs)
                        parent.values[position] = original - epsilon
                        lower = objective(layer, inputs)
                        parent.values[position] = original
                        self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)
                self.assertTrue(all(isfinite(value) for value in layer.weight.grad.values))
                axes = [name for row in layer.rotations for name in row]
                for name in ('RX', 'RY', 'RZ'):
                    self.assertTrue(any(abs(value) > 1e-7 for axis, value in zip(axes, layer.weight.grad.values, strict=True) if axis == name))

    def test_batch_accumulates_shared_weight_gradients(self):
        inputs = [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8]]
        batch, single = qnn.PQC(2, seed=42), qnn.PQC(2, seed=42)
        batch(inputs).sum().backward()
        for sample in inputs: single(sample).sum().backward()
        self.assertValues(batch.weight.grad.values, single.weight.grad.values)

    def test_reusing_pqc_twice_collects_one_weight_and_accumulates_gradients(self):
        layer = qnn.PQC(2, depth=1, seed=7)
        model = nn.Model([layer, layer])
        self.assertEqual(model.parameters(), (layer.weight,))
        inputs = Tensor([[0.3, -0.5]], requires_grad=True)
        model(inputs).square().mean().backward()
        epsilon = 1e-6
        for index, analytic in enumerate(layer.weight.grad.values):
            original = layer.weight.values[index]
            layer.weight.values[index] = original + epsilon
            upper = objective(model, inputs)
            layer.weight.values[index] = original - epsilon
            lower = objective(model, inputs)
            layer.weight.values[index] = original
            self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)

    def test_classical_quantum_classical_all_parameters_match_numerical_gradients(self):
        model = hybrid_model('cz')
        inputs = Tensor([[0.2, -0.3], [0.6, 0.1]], requires_grad=True)
        model(inputs).square().mean().backward()
        epsilon = 1e-6
        for parent in (*model.parameters(), inputs):
            self.assertIsNotNone(parent.grad)
            for index, analytic in enumerate(parent.grad.values):
                original = parent.values[index]
                parent.values[index] = original + epsilon
                upper = objective(model, inputs)
                parent.values[index] = original - epsilon
                lower = objective(model, inputs)
                parent.values[index] = original
                self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=8)
        for layer in model.layers:
            self.assertTrue(any(abs(value) > 1e-7 for parameter in layer.parameters() for value in parameter.grad.values))

    def test_prediction_preserves_flags_gradients_and_outer_recording_context(self):
        layer = qnn.PQC(2, seed=42)
        inputs = Tensor([[0.2, -0.4]], requires_grad=True)
        layer(inputs).sum().backward()
        gradients = inputs.grad.values[:], layer.weight.grad.values[:]
        prediction = nn.Model([layer])(inputs, train=False)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertTrue(inputs.requires_grad)
        self.assertTrue(layer.weight.requires_grad)
        self.assertEqual((inputs.grad.values, layer.weight.grad.values), gradients)
        with set_recording(False): self.assertFalse(layer(inputs, train=True).requires_grad)
        self.assertTrue(is_recording())
        with self.assertRaises(ValueError): layer([0.2], train=False)
        self.assertTrue(is_recording())
        self.assertTrue(layer(inputs).requires_grad)

    def test_empty_batches_have_empty_input_and_zero_parameter_gradients(self):
        layer = qnn.PQC(2, seed=42)
        inputs = Tensor([], shape=(2, 0, 2), requires_grad=True)
        result = layer(inputs)
        self.assertEqual(result.shape, inputs.shape)
        self.assertEqual(result.values, [])
        result.sum().backward()
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertEqual(inputs.grad.values, [])
        self.assertEqual(layer.weight.grad.values, [0] * layer.weight.size)

    def test_two_updates_change_all_three_parts_and_keep_rotations_fixed(self):
        model = hybrid_model()
        optimizer = nn.SGD(model, learning_rate=0.05)
        inputs, targets = [[0.2, -0.3], [0.6, 0.1]], [[0.7], [-0.2]]
        loss_function = nn.MSE()
        original = [parameter.values[:] for parameter in model.parameters()]
        axes = model.layers[1].rotations
        initial = loss_function(model(inputs, train=False), targets).item()
        for _ in range(2):
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
        self.assertLess(loss_function(model(inputs, train=False), targets).item(), initial)
        self.assertEqual(model.layers[1].rotations, axes)
        changed = [parameter.values != previous for parameter, previous in zip(model.parameters(), original, strict=True)]
        self.assertTrue(all(changed))

    def test_network_diagram_labels_pqc_without_executing_it(self):
        layer = qnn.PQC(2, depth=3, seed=42)
        network = Network(nn.Model([nn.Linear(2, 2, seed=7), layer, nn.Linear(2, 1, seed=11)]))
        labels = '\n'.join(network.nodes.values())
        self.assertIn('PQC', labels)
        self.assertIn('2', labels)
        self.assertIn('3', labels)


    def test_invalid_configuration_and_input_shapes_raise_clear_errors(self):
        for qubits, depth, entangler in ((0, 1, 'cnot'), (2, 0, 'cnot'), (2, 1, 'cy')):
            with self.subTest(qubits=qubits, depth=depth, entangler=entangler), self.assertRaises(ValueError):
                qnn.PQC(qubits, depth=depth, entangler=entangler)
        layer = qnn.PQC(2, seed=42)
        for data in (0.2, [0.2], [0.2, 0.3, 0.4]):
            with self.subTest(data=data), self.assertRaises(ValueError): layer(data)
        with self.assertRaises(ValueError): layer.circuit([[0.2, 0.3]])


if __name__ == '__main__':
    unittest.main()
