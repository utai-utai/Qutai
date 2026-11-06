"""直接拟合两比特概率分布的两步实验与保留图表。"""

import ast
import json
import unittest
from math import cos, sin
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from qutai import nn
from tests.notebook_helpers import run_notebook


class ProbabilityNotebookTests(unittest.TestCase):
    def test_exactly_two_sgd_updates_match_analytic_probability_loss(self):
        with patch.object(nn.SGD, 'step', autospec=True, side_effect=nn.SGD.step) as step:
            result = run_notebook('quantum_probabilities')
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result['targets'], [0.15, 0, 0, 0.85])
        self.assertAlmostEqual(sum(result['targets']), 1, places=12)
        self.assertEqual([row[0] for row in result['history']], [0, 1, 2])
        angle = 1.0
        rate = result['optimizer'].learning_rate
        self.assertEqual(rate, 1.0)
        for position, (_, current_angle, loss, probability) in enumerate(result['history']):
            if position:
                previous_probability = sin(angle / 2) ** 2
                angle -= rate * (previous_probability - 0.85) * sin(angle) / 2
            expected_probability = sin(angle / 2) ** 2
            self.assertAlmostEqual(current_angle, angle, places=12)
            self.assertAlmostEqual(probability, expected_probability, places=12)
            self.assertAlmostEqual(loss, 0.5 * (expected_probability - 0.85) ** 2, places=12)
        self.assertAlmostEqual(result['initial_loss'], result['history'][0][2], places=12)
        self.assertAlmostEqual(result['angle'].item(), angle, places=12)
        self.assertTrue(all(current[2] < previous[2] for previous, current in zip(result['history'], result['history'][1:])))
        self.assertIsNotNone(result['angle'].grad)
        self.assertIs(result['optimizer'].parameters[0], result['angle'])

    def test_final_quantum_state_and_tensor_prediction_are_consistent(self):
        result = run_notebook('quantum_probabilities')
        angle = result['angle'].item()
        expected = [cos(angle / 2) ** 2, 0, 0, sin(angle / 2) ** 2]
        self.assertEqual(result['prediction'].shape, (4,))
        self.assertEqual(result['circuit'].parameters(), (result['angle'],))
        for actual in (result['prediction'].tolist(), result['state'].probabilities()):
            for value, target in zip(actual, expected, strict=True):
                self.assertAlmostEqual(value, target, places=12)
            self.assertAlmostEqual(sum(actual), 1, places=12)
        self.assertGreater(expected[3], sin(1 / 2) ** 2)

    def test_notebook_keeps_valid_svg_outputs_and_saves_no_extra_files(self):
        path = Path(__file__).resolve().parents[1] / 'examples/quantum_probabilities.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        drawings = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            tree = ast.parse(''.join(cell['source']))
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
        self.assertGreaterEqual(len(drawings), 4)


if __name__ == '__main__': unittest.main()
