"""基准使用同一输入、预热、交替顺序和中位数；不对墙钟加速比作断言。"""

import random
import unittest
from unittest.mock import patch

from examples.adjoint_benchmark import build_case, run_case
from qutai.quantum import Circuit


class AdjointBenchmarkTests(unittest.TestCase):
    def test_case_is_reproducible_without_changing_global_random_state(self):
        before = random.getstate()
        first, weight, cost = build_case(3, 2)
        second, other, _ = build_case(3, 2)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(weight.shape, (2, 3))
        self.assertEqual(weight.values, other.values)
        self.assertEqual([(gate.name, gate.wires) for gate in first.gates],
                         [(gate.name, gate.wires) for gate in second.gates])
        self.assertAlmostEqual(first.run().expectation(cost), second.run().expectation(cost), places=12)

    def test_timing_excludes_warmups_and_alternates_method_order(self):
        durations = [100, 200, 0.5, 2, 4, 0.3, 0.4, 3]
        clock = [instant for index, duration in enumerate(durations) for instant in (index * 1000, index * 1000 + duration)]
        with patch('examples.adjoint_benchmark.perf_counter', side_effect=clock), \
                patch.object(Circuit, 'expectation', autospec=True, side_effect=Circuit.expectation) as expectation:
            result = run_case(2, 1, repeats=3)
        self.assertEqual([call.kwargs['method'] for call in expectation.call_args_list],
                         ['adjoint', 'parameter-shift', 'adjoint', 'parameter-shift',
                          'parameter-shift', 'adjoint', 'adjoint', 'parameter-shift'])
        self.assertAlmostEqual(result['adjoint_seconds'], 0.4)
        self.assertAlmostEqual(result['parameter_shift_seconds'], 3)
        self.assertAlmostEqual(result['speedup'], 7.5)
        self.assertLess(result['max_gradient_error'], 1e-10)
        self.assertEqual(result['rotations'], 2)
        self.assertEqual(result['repeats'], 3)


if __name__ == '__main__': unittest.main()
