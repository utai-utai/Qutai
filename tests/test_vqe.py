import unittest
from math import cos, pi, sin, sqrt

from qutai import Tensor
from qutai.quantum import CNOT, RX, RY, Circuit, X, Z


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




if __name__ == '__main__': unittest.main()
