"""量子态的位序、局部门、概率、独立采样与测量坍缩。"""

import json
import random
import unittest
from cmath import exp
from math import cos, pi, sin, sqrt
from pathlib import Path
from xml.etree import ElementTree

from qutai import quantum as q
from tests.notebook_helpers import run_notebook


def reference_gate(amplitudes, qubits, wires, matrix):
    """按输入/输出基态展开小矩阵，独立核对局部门的下标。"""
    result = [0j] * len(amplitudes)
    for source, amplitude in enumerate(amplitudes):
        column = 0
        for wire in wires:
            column = 2 * column + ((source >> (qubits - wire - 1)) & 1)
        for row, coefficients in enumerate(matrix):
            target = source
            for index, wire in enumerate(wires):
                shift = qubits - wire - 1
                bit = (row >> (len(wires) - index - 1)) & 1
                target = (target & ~(1 << shift)) | (bit << shift)
            result[target] += coefficients[column] * amplitude
    return result


class QuantumStateTests(unittest.TestCase):
    def assertAmplitudes(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected, strict=True):
            self.assertAlmostEqual(value.real, complex(target).real, places=12)
            self.assertAlmostEqual(value.imag, complex(target).imag, places=12)

    def test_zero_state_and_qubit_zero_is_leftmost(self):
        state = q.State(3)
        self.assertEqual(state.qubits, 3)
        self.assertEqual(state.amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertIs(state.apply(q.X(0)), state)
        self.assertEqual(state.probabilities(), [0, 0, 0, 0, 1, 0, 0, 0])
        state.apply(q.X(2))
        self.assertEqual(state.sample(4, seed=42), ['101'] * 4)

    def test_bell_circuit_and_fresh_runs(self):
        circuit = q.Circuit(2, [q.H(0), q.CNOT(0, 1)])
        state = circuit.run()
        self.assertAmplitudes(state.amplitudes, [1 / sqrt(2), 0, 0, 1 / sqrt(2)])
        self.assertAmplitudes(state.probabilities(), [0.5, 0, 0, 0.5])
        self.assertIsNot(state, circuit.run())
        state.apply(q.X(0))
        self.assertAmplitudes(circuit.run().amplitudes, [1 / sqrt(2), 0, 0, 1 / sqrt(2)])

    def test_circuit_copies_input_and_append_returns_circuit(self):
        initial = q.State(2).apply(q.X(1))
        before = initial.amplitudes[:]
        circuit = q.Circuit(2)
        self.assertIs(circuit.append(q.CNOT(1, 0)), circuit)
        result = circuit.run(initial)
        self.assertEqual(initial.amplitudes, before)
        self.assertIsNot(result, initial)
        self.assertEqual(result.sample(1), ['11'])
        copied = initial.copy()
        copied.apply(q.X(0))
        self.assertEqual(initial.amplitudes, before)

    def test_nonadjacent_reversed_controls_and_swap(self):
        state = q.State(3).apply(q.X(2))
        state.apply(q.CNOT(2, 0))
        self.assertEqual(state.sample(1), ['101'])
        state.apply(q.SWAP(0, 1))
        self.assertEqual(state.sample(1), ['011'])
        state.apply(q.CNOT(0, 2))
        self.assertEqual(state.sample(1), ['011'])
        state.apply(q.CNOT(1, 2))
        self.assertEqual(state.sample(1), ['010'])

    def test_complex_phases_and_pauli_y(self):
        self.assertAmplitudes(q.State(1).apply(q.Y(0)).amplitudes, [0, 1j])
        self.assertAmplitudes(q.State(1).apply(q.X(0)).apply(q.Y(0)).amplitudes, [-1j, 0])
        state = q.State(1, [1 / sqrt(2), 1j / sqrt(2)])
        state.apply(q.Z(0)).apply(q.S(0)).apply(q.T(0))
        self.assertAmplitudes(state.amplitudes, [1 / sqrt(2), exp(1j * pi / 4) / sqrt(2)])
        controlled = q.State(2, [0, 0, 1 / sqrt(2), 1 / sqrt(2)])
        controlled.apply(q.CZ(1, 0))
        self.assertAmplitudes(controlled.amplitudes, [0, 0, 1 / sqrt(2), -1 / sqrt(2)])

    def test_interference_and_bell_uncomputation(self):
        state = q.Circuit(1, [q.H(0), q.Z(0), q.H(0)]).run()
        self.assertAmplitudes(state.amplitudes, [0, 1])
        state = q.Circuit(2, [q.H(0), q.CNOT(0, 1), q.CNOT(0, 1), q.H(0)]).run()
        self.assertAmplitudes(state.amplitudes, [1, 0, 0, 0])

    def test_rotation_formulas_and_inverse(self):
        angle = 0.73
        c, s = cos(angle / 2), sin(angle / 2)
        for gate, expected in ((q.RX(0, angle), [c, -1j * s]),
                               (q.RY(0, angle), [c, s]),
                               (q.RZ(0, angle), [exp(-1j * angle / 2), 0])):
            with self.subTest(gate=gate.name):
                self.assertAmplitudes(q.State(1).apply(gate).amplitudes, expected)
        initial = [1 / sqrt(5), 2j / sqrt(5)]
        for rotation in (q.RX, q.RY, q.RZ):
            state = q.State(1, initial)
            state.apply(rotation(0, angle)).apply(rotation(0, -angle))
            self.assertAmplitudes(state.amplitudes, initial)

    def test_all_basic_gates_preserve_norm_with_complex_input(self):
        initial = [complex(index - 3, 4 - index) for index in range(8)]
        length = sqrt(sum(abs(value) ** 2 for value in initial))
        initial = [value / length for value in initial]
        gates = [q.I(1), q.H(2), q.X(0), q.Y(1), q.Z(2), q.S(0), q.T(1),
                 q.RX(2, 0.4), q.RY(1, -0.8), q.RZ(0, 1.1),
                 q.CNOT(2, 0), q.CZ(1, 0), q.SWAP(0, 2)]
        state = q.State(3, initial)
        for gate in gates:
            state.apply(gate)
            self.assertAlmostEqual(sum(state.probabilities()), 1.0, places=12)

    def test_generic_two_wire_matrix_matches_independent_reference(self):
        # 4 点傅里叶门包含复相位，且不等于几个单比特门的简单并排。
        matrix = tuple(tuple(1j ** (row * column) / 2 for column in range(4)) for row in range(4))
        initial = [complex(index + 1, index % 3 - 1) for index in range(16)]
        length = sqrt(sum(abs(value) ** 2 for value in initial))
        initial = [value / length for value in initial]
        for wires in ((0, 3), (3, 0), (2, 1)):
            with self.subTest(wires=wires):
                gate = q.Gate('Fourier', wires, matrix)
                actual = q.State(4, initial).apply(gate)
                self.assertAmplitudes(actual.amplitudes, reference_gate(initial, 4, wires, matrix))

    def test_marginals_preserve_requested_wire_order(self):
        probabilities = [0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20]
        state = q.State(3, [sqrt(value) for value in probabilities])
        self.assertAmplitudes(state.probabilities(0), [0.5, 0.5])
        self.assertAmplitudes(state.probabilities(2), [0.4, 0.6])
        self.assertAmplitudes(state.probabilities((2, 0)), [0.2, 0.2, 0.3, 0.3])
        definite = q.State(3).apply(q.X(0))
        self.assertEqual(definite.sample(3, wires=(2, 0), seed=1), ['01'] * 3)

    def test_seeded_sampling_is_independent_and_keeps_state_and_global_random(self):
        state = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run()
        before = state.amplitudes[:]
        random_before = random.getstate()
        samples = state.sample(400, seed=42)
        self.assertEqual(samples, state.sample(400, seed=42))
        self.assertEqual(set(samples), {'00', '11'})
        self.assertLess(abs(samples.count('00') - 200), 50)
        counts = state.counts(400, seed=42)
        self.assertEqual(counts, {outcome: samples.count(outcome) for outcome in sorted(set(samples))})
        self.assertEqual(sum(counts.values()), 400)
        self.assertEqual(state.amplitudes, before)
        self.assertEqual(random.getstate(), random_before)
        self.assertEqual(state.sample(0), [])
        self.assertEqual(state.counts(0), {})

    def test_partial_measurement_collapses_bell_correlations(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                state = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run()
                random_before = random.getstate()
                result = state.measure(0, seed=seed)
                self.assertIn(result, ('0', '1'))
                self.assertEqual(state.sample(5, seed=2), [result * 2] * 5)
                self.assertEqual(state.measure(1, seed=3), result)
                self.assertAlmostEqual(sum(state.probabilities()), 1.0, places=12)
                self.assertEqual(random.getstate(), random_before)

    def test_partial_measurement_keeps_unmeasured_superposition(self):
        state = q.Circuit(2, [q.H(0), q.H(1)]).run()
        result = state.measure(0, seed=42)
        expected = [1 / sqrt(2), 1 / sqrt(2), 0, 0] if result == '0' else [0, 0, 1 / sqrt(2), 1 / sqrt(2)]
        self.assertAmplitudes(state.amplitudes, expected)
        self.assertAmplitudes(state.probabilities(1), [0.5, 0.5])
        outcome = state.measure(seed=4)
        self.assertTrue(outcome.startswith(result))
        self.assertEqual(state.sample(3), [outcome] * 3)

    def test_invalid_states_gates_and_wire_selections(self):
        for qubits, amplitudes in ((0, None), (-1, None), (2, [1, 0]), (1, [1, 1]), (1, [0, 0])):
            with self.subTest(qubits=qubits, amplitudes=amplitudes), self.assertRaises(ValueError):
                q.State(qubits, amplitudes)
        for wires, matrix in (((0, 0), ((1, 0), (0, 1))), ((0,), ((1, 0, 0), (0, 1, 0))),
                              ((0,), ((1, 0), (0, 2)))):
            with self.subTest(wires=wires, matrix=matrix), self.assertRaises(ValueError):
                q.Gate('invalid', wires, matrix)
        state = q.State(2)
        for wires in ((), (0, 0), -1, 2, (0, 2)):
            with self.subTest(wires=wires), self.assertRaises(ValueError):
                state.probabilities(wires)
        with self.assertRaises(ValueError): state.apply(q.X(2))
        with self.assertRaises(ValueError): q.Circuit(2, [q.CNOT(0, 2)])
        with self.assertRaises(ValueError): q.Circuit(2).run(q.State(1))
        with self.assertRaises(ValueError): state.sample(-1)

    def test_bell_notebook_runs_from_start_to_finish_with_svg_outputs(self):
        result = run_notebook('bell')
        self.assertAmplitudes(result['initial'].probabilities(), [1, 0, 0, 0])
        self.assertAmplitudes(result['superposition'].probabilities(), [0.5, 0, 0.5, 0])
        self.assertAmplitudes(result['state'].probabilities(), [0.5, 0, 0, 0.5])
        self.assertEqual(sum(result['counts'].values()), 1000)
        self.assertEqual(set(result['counts']), {'00', '11'})
        self.assertEqual(result['first'], result['second'])
        expected = [1, 0, 0, 0] if result['first'] == '0' else [0, 0, 0, 1]
        self.assertAmplitudes(result['after_first'], expected)
        self.assertAmplitudes(result['collapsed'].probabilities(), expected)

        path = Path(__file__).resolve().parents[1] / 'examples' / 'bell.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        outputs = {cell['id']: cell.get('outputs', []) for cell in notebook['cells']}
        for name in ('circuit', 'probabilities', 'marginals', 'measurement'):
            drawings = [output['data']['image/svg+xml'] for output in outputs[name]
                        if 'image/svg+xml' in output.get('data', {})]
            self.assertEqual(len(drawings), 1, name)
            content = drawings[0] if isinstance(drawings[0], str) else ''.join(drawings[0])
            ElementTree.fromstring(content)


if __name__ == '__main__':
    unittest.main()
