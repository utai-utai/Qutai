"""两量子比特 VQE：已知能量、解析梯度及两步 Notebook 实验。"""

import ast
import json
import unittest
from math import cos, pi, sin, sqrt
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from qutai import Tensor, nn
from qutai.quantum import CNOT, RX, RY, Circuit, X, Z
from tests.notebook_helpers import run_notebook


class VQETests(unittest.TestCase):
    def test_trial_state_energy_and_gradient_match_analytic_formula(self):
        hamiltonian = -(X(0) @ X(1)) - 0.5 * Z(0) - 0.5 * Z(1)
        for value in (-0.4, 0.2, 0.7, pi / 4):
            with self.subTest(angle=value):
                angle = Tensor(value, requires_grad=True)
                circuit = Circuit(2, [RY(0, angle), CNOT(0, 1)])
                energy = circuit.expectation(hamiltonian)
                self.assertAlmostEqual(energy.item(), -sin(value) - cos(value), places=12)
                self.assertGreaterEqual(energy.item() + 1e-12, -sqrt(2))
                energy.backward()
                self.assertAlmostEqual(angle.grad.item(), sin(value) - cos(value), places=12)
        self.assertAlmostEqual(energy.item(), -sqrt(2), places=12)
        self.assertAlmostEqual(angle.grad.item(), 0, places=12)

    def test_energy_bound_also_holds_outside_the_trial_state_family(self):
        hamiltonian = -(X(0) @ X(1)) - 0.5 * Z(0) - 0.5 * Z(1)
        for first, second in ((0.31, 0.87), (-0.41, 1.2), (2.7, -1.1)):
            state = Circuit(2, [RY(0, first), RX(1, second), CNOT(1, 0)]).run()
            self.assertGreaterEqual(state.expectation(hamiltonian) + 1e-12, -sqrt(2))

    def test_notebook_updates_match_analytic_energy_history(self):
        with patch.object(nn.SGD, 'step', autospec=True, side_effect=nn.SGD.step) as step:
            result = run_notebook('vqe')
        self.assertGreater(step.call_count, 0)
        self.assertEqual(len(result['history']), step.call_count + 1)
        self.assertAlmostEqual(result['minimum'], -sqrt(2), places=12)
        expected_angle = 0.2
        learning_rate = result['optimizer'].learning_rate
        self.assertEqual(learning_rate, 0.2)
        for position, (_, angle, energy) in enumerate(result['history']):
            if position: expected_angle -= learning_rate * (sin(expected_angle) - cos(expected_angle))
            self.assertAlmostEqual(angle, expected_angle, places=12)
            self.assertAlmostEqual(energy, -sin(expected_angle) - cos(expected_angle), places=12)
            self.assertGreaterEqual(energy + 1e-12, result['minimum'])
        self.assertAlmostEqual(result['angle'].item(), expected_angle, places=12)
        self.assertAlmostEqual(result['initial_energy'], result['history'][0][2], places=12)
        self.assertTrue(all(current[2] < previous[2] for previous, current in zip(result['history'], result['history'][1:])))
        self.assertAlmostEqual(result['state'].expectation(result['hamiltonian']), result['history'][-1][2], places=12)
        self.assertEqual(result['circuit'].parameters(), (result['angle'],))

    def test_notebook_keeps_svg_outputs_and_does_not_save_extra_files(self):
        path = Path(__file__).resolve().parents[1] / 'examples' / 'vqe.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        drawings = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            source = ''.join(cell['source'])
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('save', 'write_text', 'write_bytes'))
            self.assertIsNotNone(cell['execution_count'])
            for output in cell['outputs']:
                self.assertNotEqual(output['output_type'], 'error')
                drawing = output.get('data', {}).get('image/svg+xml')
                if drawing:
                    if isinstance(drawing, list): drawing = ''.join(drawing)
                    ElementTree.fromstring(drawing)
                    drawings.append(drawing)
        self.assertGreaterEqual(len(drawings), 3)


if __name__ == '__main__': unittest.main()
