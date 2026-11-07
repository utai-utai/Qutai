import unittest
from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree

from qutai import quantum as q
from qutai.visualization import Circuit, Probabilities


SVG = '{http://www.w3.org/2000/svg}'


class QuantumVisualizationTests(unittest.TestCase):
    def test_independent_rotations_share_a_column_before_cnot_chain(self):
        circuit = q.Circuit(3, [q.RY(0, 0.2), q.RX(1, 0.4), q.RZ(2, 0.6), q.CNOT(0, 1), q.CNOT(1, 2)])
        root = ElementTree.fromstring(Circuit(circuit)._repr_svg_())
        gates = [node for node in root.iter() if 'data-gate' in node.attrib]
        self.assertEqual([int(node.get('data-column')) for node in gates], [0, 0, 0, 1, 2])
        self.assertEqual(len({node.find(SVG + 'rect').get('x') for node in gates[:3]}), 1)
        self.assertLess(float(root.get('width')), 190 + len(circuit.gates) * 90)

    def test_shared_wires_keep_gate_order_while_other_wires_can_pack_earlier(self):
        circuit = q.Circuit(2, [q.H(0), q.X(0), q.RZ(1, 0.3), q.CNOT(0, 1), q.Z(1), q.Y(0)])
        root = ElementTree.fromstring(Circuit(circuit)._repr_svg_())
        gates = [node for node in root.iter() if 'data-gate' in node.attrib]
        self.assertEqual([int(node.get('data-column')) for node in gates], [0, 1, 0, 2, 3, 3])
        self.assertEqual([node.get('data-gate') for node in gates], [gate.name for gate in circuit.gates])

    def test_long_and_reversed_connectors_reserve_the_intermediate_wire(self):
        for control, target in ((0, 2), (2, 0)):
            with self.subTest(control=control, target=target):
                circuit = q.Circuit(3, [q.H(1), q.CNOT(control, target), q.X(1)])
                root = ElementTree.fromstring(Circuit(circuit)._repr_svg_())
                gates = [node for node in root.iter() if 'data-gate' in node.attrib]
                self.assertEqual([int(node.get('data-column')) for node in gates], [0, 1, 2])

    def test_circuit_draws_without_simulation_and_escapes_title(self):
        circuit = q.Circuit(2, [q.H(0), q.CNOT(0, 1)])
        with patch.object(q.Circuit, 'run', side_effect=AssertionError('画电路不应执行模拟')):
            content = Circuit(circuit, title='<Bell> & "态"')._repr_svg_()
        root = ElementTree.fromstring(content)
        self.assertEqual(root.find(SVG + 'title').text, '<Bell> & "态"')
        self.assertIn('&lt;Bell&gt;', content)
        gates = [node for node in root.iter() if 'data-gate' in node.attrib]
        self.assertEqual([(node.get('data-gate'), node.get('data-wires')) for node in gates],
                         [('H', '0'), ('CNOT', '0,1')])
        wires = [node.get('data-wire') for node in root.iter() if 'data-wire' in node.attrib]
        self.assertEqual(wires, ['0', '1'])
        self.assertIn('q0', ''.join(root.itertext()))
        self.assertIn('q1', ''.join(root.itertext()))

    def test_controls_reversed_wires_swap_and_rotation_labels(self):
        circuit = q.Circuit(3, [q.CNOT(2, 0), q.CZ(1, 2), q.SWAP(0, 2), q.RY(1, pi / 3)])
        root = ElementTree.fromstring(Circuit(circuit)._repr_svg_())
        roles = [node.get('data-role') for node in root.iter()]
        self.assertEqual(roles.count('control'), 3)
        self.assertEqual(roles.count('target'), 1)
        self.assertEqual(roles.count('swap'), 2)
        gates = [node for node in root.iter() if 'data-gate' in node.attrib]
        self.assertEqual([node.get('data-wires') for node in gates], ['2,0', '1,2', '0,2', '1'])
        self.assertIn('RY', ''.join(root.itertext()))

    def test_custom_multiwire_gate_and_empty_circuit_are_valid_svg(self):
        custom = q.Gate('<U&>', (2, 0), q.SWAP(0, 1).matrix)
        for circuit in (q.Circuit(3, [custom]), q.Circuit(1)):
            with self.subTest(gates=len(circuit.gates)):
                content = Circuit(circuit)._repr_svg_()
                root = ElementTree.fromstring(content)
                if circuit.gates:
                    gate = next(node for node in root.iter() if 'data-gate' in node.attrib)
                    self.assertEqual(gate.get('data-gate'), '<U&>')
                    self.assertEqual(gate.get('data-wires'), '2,0')
                self.assertGreater(float(root.get('width')), 0)
                self.assertGreater(float(root.get('height')), 0)

    def test_single_wire_custom_gate_names_do_not_imply_two_wire_symbols(self):
        for name in ('CNOT', 'CZ', 'SWAP'):
            with self.subTest(name=name):
                gate = q.Gate(name, (0,), q.H(0).matrix)
                content = Circuit(q.Circuit(1, [gate]))._repr_svg_()
                root = ElementTree.fromstring(content)
                self.assertIn(name, ''.join(root.itertext()))
                self.assertFalse(any(node.get('data-role') in ('control', 'target', 'swap') for node in root.iter()))

    def test_probabilities_include_zero_bars_in_computational_order(self):
        state = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run()
        before = state.amplitudes[:]
        content = Probabilities(state, title='Bell 概率')._repr_svg_()
        root = ElementTree.fromstring(content)
        bars = [node for node in root.iter(SVG + 'rect') if 'data-basis' in node.attrib]
        self.assertEqual([bar.get('data-basis') for bar in bars], ['00', '01', '10', '11'])
        for bar, expected in zip(bars, (0.5, 0, 0, 0.5), strict=True):
            self.assertAlmostEqual(float(bar.get('data-probability')), expected)
            if expected == 0: self.assertEqual(float(bar.get('height')), 0)
        labels = ''.join(root.itertext())
        for basis in ('00', '01', '10', '11'):
            self.assertIn(f'|{basis}⟩', labels)
        self.assertEqual(state.amplitudes, before)

    def test_probability_marginal_respects_requested_wire_order(self):
        state = q.State(3).apply(q.X(0))
        root = ElementTree.fromstring(Probabilities(state, wires=(2, 0))._repr_svg_())
        values = {node.get('data-basis'): float(node.get('data-probability'))
                  for node in root.iter(SVG + 'rect') if 'data-basis' in node.attrib}
        self.assertEqual(values, {'00': 0, '01': 1, '10': 0, '11': 0})
        labels = ''.join(root.itertext())
        self.assertIn('q2', labels)
        self.assertIn('q0', labels)

    def test_both_views_save_standalone_svg(self):
        circuit = q.Circuit(2, [q.H(0), q.CNOT(0, 1)])
        for name, view in (('circuit', Circuit(circuit)), ('probabilities', Probabilities(circuit.run()))):
            with self.subTest(view=name), TemporaryDirectory() as folder:
                target = Path(folder) / f'{name}.svg'
                self.assertEqual(view.save(target), target)
                saved = target.read_text(encoding='utf-8')
                self.assertEqual(saved, view._repr_svg_())
                ElementTree.fromstring(saved)
                self.assertNotIn('http://', saved.replace('http://www.w3.org/2000/svg', ''))
                self.assertIn('background:white', saved)


if __name__ == '__main__':
    unittest.main()
