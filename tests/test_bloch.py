import unittest
from cmath import exp
from math import cos, pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from qutai import is_available, quantum as q
from qutai.visualization import Bloch


SVG = '{http://www.w3.org/2000/svg}'


class BlochTests(unittest.TestCase):
    def assertVector(self, view, expected):
        for value, target in zip(view.vector, expected, strict=True):
            self.assertAlmostEqual(value, target, places=12)

    def test_six_cardinal_states_have_correct_pauli_directions(self):
        cases = [([], (0, 0, 1)), ([q.X(0)], (0, 0, -1)),
                 ([q.H(0)], (1, 0, 0)), ([q.H(0), q.Z(0)], (-1, 0, 0)),
                 ([q.H(0), q.S(0)], (0, 1, 0)), ([q.H(0), q.S(0), q.Z(0)], (0, -1, 0))]
        for gates, expected in cases:
            with self.subTest(expected=expected):
                state = q.Circuit(1, gates).run()
                before = state.amplitudes[:]
                view = Bloch(state)
                self.assertVector(view, expected)
                self.assertAlmostEqual(view.length, 1)
                self.assertEqual(state.amplitudes, before)

    def test_arbitrary_complex_state_matches_analytic_bloch_vector(self):
        theta, phi = 0.8, -1.2
        state = q.State(1, [cos(theta / 2), exp(1j * phi) * sin(theta / 2)])
        self.assertVector(Bloch(state), (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)))

    def test_bell_qubit_is_at_center_without_normalizing_vector(self):
        state = q.Circuit(2, [q.H(0), q.CNOT(0, 1)]).run()
        for wire in (0, 1):
            with self.subTest(wire=wire):
                view = Bloch(state, wire)
                self.assertVector(view, (0, 0, 0))
                self.assertAlmostEqual(view.length, 0)
                root = ElementTree.fromstring(view._repr_svg_())
                point = next(node for node in root.iter() if node.get('data-role') == 'bloch-point')
                self.assertEqual((float(point.get('cx')), float(point.get('cy'))), (240, 235))
                self.assertFalse(any(node.get('data-role') == 'bloch-vector' for node in root.iter()))
                self.assertIn('最大混合态', ''.join(root.itertext()))

    def test_partially_entangled_reduced_state_stays_inside_sphere(self):
        angle = pi / 7
        state = q.State(2, [cos(angle), 0, 0, sin(angle)])
        for wire in (0, 1):
            view = Bloch(state, wire)
            self.assertVector(view, (0, 0, cos(2 * angle)))
            self.assertAlmostEqual(view.length, cos(2 * angle))
            self.assertGreater(view.length, 0)
            self.assertLess(view.length, 1)

    def test_complex64_labels_preserve_pure_and_mixed_state_vectors(self):
        cases = [("纯态", [q.H(0)], True),
                 ("最大混合态", [q.H(0), q.CNOT(0, 1)], False),
                 ("混合态", [q.RY(0, 0.8), q.CNOT(0, 1)], True)]
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                if not is_available(device): self.skipTest(f"需要实际 {device} 设备")
                for kind, gates, arrow in cases:
                    with self.subTest(kind=kind):
                        state = q.Circuit(2, gates).run(q.State(2, device=device, dtype="complex64"))
                        before = state.storage.tolist()
                        vector = tuple(state.expectation([q.X(0), q.Y(0), q.Z(0)]))
                        view = Bloch(state)
                        root = ElementTree.fromstring(view._repr_svg_())
                        content = ''.join(root.itertext())
                        self.assertIn(f"· {kind}", content)
                        self.assertNotIn("-0.000", content)
                        self.assertEqual(any(node.get('data-role') == 'bloch-vector' for node in root.iter()), arrow)
                        self.assertEqual(view.vector, vector)
                        self.assertEqual(state.storage.tolist(), before)

    def test_wire_zero_is_leftmost_and_view_is_snapshot(self):
        state = q.State(2).apply(q.X(0))
        view = Bloch(state, 0)
        self.assertVector(view, (0, 0, -1))
        self.assertVector(Bloch(state, 1), (0, 0, 1))
        before = view._repr_svg_()
        state.apply(q.X(0))
        self.assertEqual(view._repr_svg_(), before)

    def test_svg_projection_axes_and_state_points_agree(self):
        roots = {}
        for name, gates in (('z', []), ('x', [q.H(0)]), ('y', [q.H(0), q.S(0)])):
            roots[name] = ElementTree.fromstring(Bloch(q.Circuit(1, gates).run())._repr_svg_())
        points = {name: next(node for node in root.iter() if node.get('data-role') == 'bloch-point')
                  for name, root in roots.items()}
        self.assertLess(float(points['z'].get('cy')), 235)
        self.assertLess(float(points['x'].get('cx')), 240)
        self.assertGreater(float(points['x'].get('cy')), 235)
        self.assertGreater(float(points['y'].get('cx')), 240)
        self.assertGreater(float(points['y'].get('cy')), 235)
        self.assertEqual({node.get('data-axis') for node in roots['z'].iter() if 'data-axis' in node.attrib}, {'x', 'y', 'z'})

    def test_invalid_wire_is_rejected(self):
        for wire in (-1, 2):
            with self.subTest(wire=wire), self.assertRaises(ValueError): Bloch(q.State(2), wire)
        for wire in (0.0, (0, 1), None):
            with self.subTest(wire=wire), self.assertRaises(TypeError): Bloch(q.State(2), wire)

    def test_save_standalone_svg_and_escape_title(self):
        view = Bloch(q.State(1), title='<Bloch> & "状态"')
        content = view._repr_svg_()
        root = ElementTree.fromstring(content)
        self.assertEqual(root.find(SVG + 'title').text, '<Bloch> & "状态"')
        self.assertIn('|0⟩', ''.join(root.itertext()))
        self.assertIn('|1⟩', ''.join(root.itertext()))
        self.assertNotIn('http://', content.replace('http://www.w3.org/2000/svg', ''))
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'bloch.svg'
            self.assertEqual(view.save(target), target)
            self.assertEqual(target.read_text(encoding='utf-8'), content)


if __name__ == '__main__':
    unittest.main()
