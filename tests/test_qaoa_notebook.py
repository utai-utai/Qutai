"""四节点 MaxCut：两步 QAOA、独立数值轨迹和可视化输出。"""

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from qutai import nn
from qutai.quantum import QAOA, maxcut
from tests.notebook_helpers import run_notebook
from tests.test_qaoa import reference_energy


class QAOANotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(nn.SGD, 'step', autospec=True, side_effect=nn.SGD.step) as step:
            cls.result = run_notebook('qaoa')
        cls.updates = step.call_count

    def test_updates_match_independent_numerical_gradient_trajectory(self):
        result = self.result
        self.assertGreater(self.updates, 0)
        self.assertEqual(len(result['history']), self.updates + 1)
        initial = QAOA(result['qubits'], maxcut(result['edges']), depth=1, seed=42)
        weights = initial.weight.values[:]
        eigenvalues = [-value for value in result['cut_values']]
        learning_rate = result['optimizer'].learning_rate
        self.assertEqual(learning_rate, 0.1)
        epsilon = 1e-6
        for position, (_, energy, expected_cut) in enumerate(result['history']):
            if position:
                derivatives = []
                for index in range(len(weights)):
                    upper, lower = weights[:], weights[:]
                    upper[index] += epsilon
                    lower[index] -= epsilon
                    derivatives.append((reference_energy(result['qubits'], eigenvalues, upper)
                                        - reference_energy(result['qubits'], eigenvalues, lower)) / (2 * epsilon))
                weights = [value - learning_rate * derivative for value, derivative in zip(weights, derivatives, strict=True)]
            self.assertAlmostEqual(energy, reference_energy(result['qubits'], eigenvalues, weights), places=8)
            self.assertAlmostEqual(expected_cut, -energy, places=12)
            self.assertGreaterEqual(energy + 1e-12, -result['maximum'])
            self.assertLessEqual(energy, 1e-12)
        self.assertAlmostEqual(result['initial_energy'], result['history'][0][1], places=12)
        self.assertTrue(all(current[1] < previous[1] for previous, current in zip(result['history'], result['history'][1:])))
        for actual, expected in zip(result['algorithm'].weight.values, weights, strict=True):
            self.assertAlmostEqual(actual, expected, places=8)
        self.assertNotEqual(result['algorithm'].weight.values, initial.weight.values)

    def test_probabilities_and_reported_cut_agree_with_exhaustive_small_graph(self):
        result = self.result
        self.assertEqual(result['qubits'], 4)
        self.assertEqual(result['bitstrings'], [f'{index:04b}' for index in range(16)])
        actual_cuts = [sum(bits[first] != bits[second] for first, second in result['edges']) for bits in result['bitstrings']]
        self.assertEqual(result['cut_values'], actual_cuts)
        self.assertEqual(result['maximum'], max(actual_cuts))
        self.assertEqual(result['maximum'], 4)
        self.assertEqual(result['optimal_states'], [bits for bits, cut in zip(result['bitstrings'], actual_cuts) if cut == 4])
        self.assertEqual(result['optimal_states'], ['0101', '1010'])
        probabilities = result['probabilities']
        self.assertAlmostEqual(sum(probabilities), 1, places=12)
        for actual, expected in zip(probabilities, result['state'].probabilities(), strict=True):
            self.assertAlmostEqual(actual, expected, places=12)
        expected_cut = sum(probability * cut for probability, cut in zip(probabilities, actual_cuts, strict=True))
        self.assertAlmostEqual(expected_cut, result['history'][-1][2], places=12)
        self.assertAlmostEqual(result['state'].expectation(result['cost']), -expected_cut, places=12)
        self.assertEqual(sum(result['counts'].values()), 1000)
        self.assertTrue(set(result['counts']).issubset(result['bitstrings']))
        self.assertEqual(result['best_cut'], actual_cuts[result['bitstrings'].index(result['best'])])
        self.assertEqual(result['best_cut'], result['maximum'])
        self.assertAlmostEqual(probabilities[result['bitstrings'].index(result['best'])], max(probabilities), places=12)
        self.assertEqual(result['best_partition'], {wire: int(bit) for wire, bit in enumerate(result['best'])})

    def test_notebook_retains_valid_drawings_and_saves_no_extra_files(self):
        path = Path(__file__).resolve().parents[1] / 'examples' / 'qaoa.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len({cell['id'] for cell in notebook['cells']}), len(notebook['cells']))
        drawings = []
        graph_calls = 0
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            self.assertIsNotNone(cell['execution_count'])
            tree = ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('save', 'write_text', 'write_bytes'))
                    graph_calls += node.func.attr == 'Graph'
            for output in cell['outputs']:
                self.assertNotEqual(output['output_type'], 'error')
                drawing = output.get('data', {}).get('image/svg+xml')
                if drawing:
                    if isinstance(drawing, list): drawing = ''.join(drawing)
                    ElementTree.fromstring(drawing)
                    drawings.append(drawing)
        self.assertGreaterEqual(graph_calls, 2)
        self.assertGreaterEqual(len(drawings), 5)


if __name__ == '__main__': unittest.main()
