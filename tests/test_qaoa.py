"""QAOA 成本相位、混合演化、共享角度梯度与两步优化。"""

import random
import unittest
from cmath import exp
from math import cos, pi, sin, sqrt

from qutai import Tensor, nn
from qutai.core.context import set_recording
from qutai.quantum import H, I, QAOA, State, X, Y, Z, maxcut


def reference_state(qubits, eigenvalues, angles):
    """直接按每个基态的能量加相位，再逐位计算 X 混合；不使用门或 CPU 内核。"""
    amplitudes = [1 / sqrt(2 ** qubits) + 0j] * (2 ** qubits)
    for gamma, beta in angles:
        amplitudes = [amplitude * exp(-1j * gamma * value)
                      for amplitude, value in zip(amplitudes, eigenvalues, strict=True)]
        for wire in range(qubits):
            mask = 1 << (qubits - wire - 1)
            amplitudes = [cos(beta) * value - 1j * sin(beta) * amplitudes[index ^ mask]
                          for index, value in enumerate(amplitudes)]
    return amplitudes


def reference_energy(qubits, eigenvalues, weights):
    angles = list(zip(weights[::2], weights[1::2], strict=True))
    amplitudes = reference_state(qubits, eigenvalues, angles)
    return sum(abs(amplitude) ** 2 * value for amplitude, value in zip(amplitudes, eigenvalues, strict=True))


def z_eigenvalue(index, qubits, wire):
    return 1 - 2 * ((index >> (qubits - wire - 1)) & 1)


def interacting_cost():
    return (1.25 * I(1) - 0.6 * Z(0) + 0.4 * Z(2)
            + 0.7 * (Z(2) @ Z(0)) - 0.3 * (Z(2) @ Z(0) @ Z(1)))


def interacting_eigenvalues():
    result = []
    for index in range(8):
        first, second, third = [z_eigenvalue(index, 3, wire) for wire in range(3)]
        result.append(1.25 - 0.6 * first + 0.4 * third + 0.7 * third * first - 0.3 * third * first * second)
    return result


class QAOATests(unittest.TestCase):
    def assertSameState(self, actual, expected):
        pivot = max(range(len(expected)), key=lambda index: abs(expected[index]))
        phase = actual[pivot] / expected[pivot]
        self.assertAlmostEqual(abs(phase), 1, places=11)
        for left, right in zip(actual, expected, strict=True):
            self.assertAlmostEqual(abs(left - phase * right), 0, places=11)

    def test_maxcut_cost_is_negative_cut_for_every_basis_state(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        cost = maxcut(edges)
        for index in range(8):
            bits = f'{index:03b}'
            amplitudes = [0j] * 8
            amplitudes[index] = 1
            cut = sum(bits[first] != bits[second] for first, second in edges)
            self.assertAlmostEqual(State(3, amplitudes).expectation(cost), -cut, places=12)
        self.assertAlmostEqual(State(2).expectation(maxcut(iter([(1, 0)]))), 0, places=12)

    def test_state_and_energy_match_independent_full_basis_reference(self):
        algorithm = QAOA(3, interacting_cost(), depth=2, seed=7)
        algorithm.weight.values[:] = [0.23, -0.37, 0.51, 0.19]
        eigenvalues = interacting_eigenvalues()
        expected = reference_state(3, eigenvalues, [(0.23, -0.37), (0.51, 0.19)])
        state = algorithm.run()
        self.assertSameState(state.amplitudes, expected)
        self.assertAlmostEqual(sum(state.probabilities()), 1, places=12)
        self.assertAlmostEqual(algorithm.expectation().item(), reference_energy(3, eigenvalues, algorithm.weight.values), places=12)

    def test_every_gamma_and_beta_gradient_matches_independent_central_difference(self):
        algorithm = QAOA(3, interacting_cost(), depth=2, seed=7)
        initial = [0.23, -0.37, 0.51, 0.19]
        algorithm.weight.values[:] = initial
        algorithm.expectation().backward()
        self.assertEqual(algorithm.weight.grad.shape, (2, 2))
        epsilon = 1e-6
        for index, derivative in enumerate(algorithm.weight.grad.values):
            upper, lower = initial[:], initial[:]
            upper[index] += epsilon
            lower[index] -= epsilon
            numeric = (reference_energy(3, interacting_eigenvalues(), upper)
                       - reference_energy(3, interacting_eigenvalues(), lower)) / (2 * epsilon)
            self.assertAlmostEqual(derivative, numeric, places=8)

    def test_single_edge_analytic_formula_checks_sign_and_factors_of_two(self):
        algorithm = QAOA(2, maxcut([(0, 1)]), seed=3)
        gamma, beta = 0.7, -0.2
        algorithm.weight.values[:] = [gamma, beta]
        energy = algorithm.expectation()
        self.assertAlmostEqual(energy.item(), (sin(4 * beta) * sin(gamma) - 1) / 2, places=12)
        energy.backward()
        self.assertAlmostEqual(algorithm.weight.grad.values[0], sin(4 * beta) * cos(gamma) / 2, places=12)
        self.assertAlmostEqual(algorithm.weight.grad.values[1], 2 * cos(4 * beta) * sin(gamma), places=12)
        algorithm.weight.values[:] = [pi / 2, -pi / 8]
        self.assertAlmostEqual(algorithm.expectation().item(), -1, places=12)
        self.assertAlmostEqual(sum(algorithm.run().probabilities()[index] for index in (1, 2)), 1, places=12)

    def test_local_random_seed_does_not_change_global_random_state(self):
        before = random.getstate()
        first = QAOA(2, maxcut([(0, 1)]), depth=3, seed=42)
        second = QAOA(2, maxcut([(0, 1)]), depth=3, seed=42)
        QAOA(2, maxcut([(0, 1)]))
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first.weight.values, second.weight.values)
        self.assertEqual(first.weight.shape, (3, 2))
        self.assertTrue(first.weight.requires_grad)
        self.assertEqual(first.parameters(), (first.weight,))
        self.assertIs(nn.SGD(first).parameters[0], first.weight)

    def test_current_weights_build_fresh_circuits_and_change_predictions(self):
        algorithm = QAOA(2, maxcut([(0, 1)]), seed=3)
        algorithm.weight.values[:] = [0.3, -0.2]
        first = algorithm.circuit()
        old_energy = algorithm.expectation().item()
        algorithm.weight.values[:] = [0.8, -0.35]
        second = algorithm.circuit()
        self.assertIsNot(first, second)
        self.assertNotAlmostEqual(algorithm.expectation().item(), old_energy, places=6)
        self.assertAlmostEqual(algorithm.expectation().item(), reference_energy(2, [0, -1, -1, 0], [0.8, -0.35]), places=12)

    def test_old_graph_and_input_cost_are_snapshots(self):
        cost = maxcut([(0, 1)])
        algorithm = QAOA(2, cost, seed=3)
        gamma, beta = 0.7, -0.2
        algorithm.weight.values[:] = [gamma, beta]
        old = algorithm.expectation()
        algorithm.weight.values[:] = [0.8, 0.25]
        cost.terms = ()
        old.backward()
        self.assertAlmostEqual(algorithm.weight.grad.values[0], sin(4 * beta) * cos(gamma) / 2, places=12)
        self.assertAlmostEqual(algorithm.weight.grad.values[1], 2 * cos(4 * beta) * sin(gamma), places=12)
        self.assertAlmostEqual(algorithm.expectation().item(), reference_energy(2, [0, -1, -1, 0], [0.8, 0.25]), places=12)

    def test_global_constant_changes_energy_but_not_probabilities_or_gradients(self):
        cost = maxcut([(0, 1), (1, 2)])
        first, second = QAOA(3, cost, depth=2, seed=42), QAOA(3, cost + 7.5, depth=2, seed=42)
        first_energy, second_energy = first.expectation(), second.expectation()
        self.assertAlmostEqual(second_energy.item() - first_energy.item(), 7.5, places=12)
        self.assertSameState(first.run().amplitudes, second.run().amplitudes)
        first_energy.backward()
        second_energy.backward()
        for left, right in zip(first.weight.grad.values, second.weight.grad.values, strict=True):
            self.assertAlmostEqual(left, right, places=11)

    def test_isolated_qubit_and_identity_factor_preserve_problem_distribution(self):
        cost = 0.5 * ((Z(0) @ Z(1)) @ I(2)) - 0.5
        larger = QAOA(3, cost, depth=2, seed=42)
        smaller = QAOA(2, maxcut([(0, 1)]), depth=2, seed=42)
        for actual, expected in zip(larger.run().probabilities((0, 1)), smaller.run().probabilities(), strict=True):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(larger.run().expectation(X(2)), 1, places=12)

    def test_empty_graph_has_zero_energy_and_zero_gradients(self):
        algorithm = QAOA(3, maxcut([]), depth=2, seed=42)
        energy = algorithm.expectation()
        self.assertIsInstance(energy, Tensor)
        self.assertAlmostEqual(energy.item(), 0, places=12)
        energy.backward()
        self.assertTrue(all(abs(value) < 1e-12 for value in algorithm.weight.grad.values))
        self.assertTrue(all(abs(value - 1 / 8) < 1e-12 for value in algorithm.run().probabilities()))

    def test_prediction_context_keeps_parameters_trainable(self):
        algorithm = QAOA(2, maxcut([(0, 1)]), seed=3)
        with set_recording(False):
            result = algorithm.expectation()
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.parents, ())
        self.assertTrue(algorithm.weight.requires_grad)
        self.assertTrue(algorithm.expectation().requires_grad)

    def test_two_sgd_updates_lower_the_expected_cost(self):
        algorithm = QAOA(2, maxcut([(0, 1)]), seed=3)
        algorithm.weight.values[:] = [0.3, -0.2]
        optimizer = nn.SGD(algorithm, learning_rate=0.1)
        history = [algorithm.expectation().item()]
        gamma, beta = 0.3, -0.2
        for _ in range(2):
            optimizer.zero_grad()
            energy = algorithm.expectation()
            energy.backward()
            gamma, beta = (gamma - 0.1 * sin(4 * beta) * cos(gamma) / 2,
                           beta - 0.1 * 2 * cos(4 * beta) * sin(gamma))
            optimizer.step()
            history.append(algorithm.expectation().item())
            self.assertAlmostEqual(algorithm.weight.values[0], gamma, places=12)
            self.assertAlmostEqual(algorithm.weight.values[1], beta, places=12)
        self.assertTrue(all(current < previous for previous, current in zip(history, history[1:])))

    def test_invalid_problem_and_depth_are_rejected_upfront(self):
        for qubits in (0, -1, 1.5, True):
            with self.subTest(qubits=qubits), self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                QAOA(qubits, Z(0))
        for depth in (0, -1, 1.5, True):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                QAOA(2, Z(0), depth=depth)
        for cost in (X(0), Y(0), H(0), Z(2)):
            with self.subTest(cost=cost), self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                QAOA(2, cost)
        with self.assertRaises(ValueError): maxcut([(0, 0)])


if __name__ == '__main__': unittest.main()
