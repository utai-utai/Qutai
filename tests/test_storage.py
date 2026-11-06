"""连续 CPU 存储、后端边界与复数数值内核。"""

import math
import unittest
from array import array

from qutai.backends import cpu
from qutai.backends.device import get_backend
from qutai.backends.storage import Storage


class StorageTests(unittest.TestCase):
    def test_float64_buffer_owns_values_and_exports_independent_lists(self):
        values = [1, 2, 3]
        data = cpu.storage(values)
        self.assertIsInstance(data, Storage)
        self.assertEqual((data.device, data.dtype, data.size), ("cpu", "float64", 3))
        self.assertIsInstance(data.buffer, array)
        self.assertEqual((data.buffer.typecode, data.buffer.itemsize), ("d", 8))
        self.assertTrue(memoryview(data.buffer).contiguous)
        values[0] = 999
        exported = data.tolist()
        exported[1] = 999
        self.assertEqual(data, [1, 2, 3])

    def test_complex128_uses_interleaved_doubles_and_logical_indices(self):
        data = cpu.storage([1 + 2j, 3 - 4j, 5j], dtype="complex128")
        self.assertEqual(data.size, 3)
        self.assertEqual(data.buffer.tolist(), [1, 2, 3, -4, 0, 5])
        self.assertEqual(data[-1], 5j)
        data[1] = -2 + 8j
        self.assertEqual(data.tolist(), [1 + 2j, -2 + 8j, 5j])
        self.assertEqual(data[::-1].tolist(), [5j, -2 + 8j, 1 + 2j])
        data[::2] = [9j, 7]
        self.assertEqual(data.tolist(), [9j, -2 + 8j, 7])

    def test_copies_and_slices_are_independent_but_copy_into_keeps_identity(self):
        source = cpu.storage([1, 2, 3])
        copied, sliced = source.copy(), source[:2]
        source[:] = [4, 5, 6]
        self.assertEqual(copied, [1, 2, 3])
        self.assertEqual(sliced, [1, 2])
        identity = id(copied)
        cpu.copy_into(copied, source)
        self.assertEqual(id(copied), identity)
        self.assertEqual(copied, [4, 5, 6])
        with self.assertRaises(ValueError): cpu.copy_into(copied, sliced)

    def test_empty_buffers_devices_and_precisions(self):
        for dtype in ("float64", "complex128"):
            data = cpu.storage(dtype=dtype)
            self.assertEqual((data.size, data.tolist()), (0, []))
            self.assertEqual(data.copy().dtype, dtype)
        self.assertIs(get_backend(), cpu)
        for device in ("bad", "mps", "unknown"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "设备"):
                get_backend(device)
        with self.assertRaisesRegex(ValueError, "精度"):
            cpu.storage([1], dtype="float16")

    def test_named_elementwise_kernels_produce_storage(self):
        left, right = cpu.storage([2, -3]), cpu.storage([4, 2])
        for operation, expected in ((cpu.add, [6, -1]), (cpu.subtract, [-2, -5]),
                                    (cpu.multiply, [8, -6]), (cpu.divide, [0.5, -1.5])):
            result = operation(left, right)
            self.assertIsInstance(result, Storage)
            self.assertEqual(result, expected)
        self.assertEqual(cpu.square(left), [4, 9])
        self.assertEqual(cpu.relu(left), [2, 0])
        self.assertEqual(cpu.relu_backward(right, left), [4, 0])
        self.assertEqual(cpu.negate(left), [-2, 3])
        self.assertEqual(cpu.scale(left, 0.5), [1, -1.5])
        for actual, expected in zip(cpu.log(cpu.exp(left)), left, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_axis_concatenation_keeps_batches_heads_and_complex_values(self):
        first = cpu.storage([1, 2, 3, 4], dtype="complex128")
        second = cpu.storage([5j, 6j, 7j, 8j], dtype="complex128")
        joined = cpu.concatenate((first, second), ((2, 2, 1, 1), (2, 2, 1, 1)), axis=2)
        self.assertEqual(joined.dtype, "complex128")
        self.assertEqual(joined.tolist(), [1, 5j, 2, 6j, 3, 7j, 4, 8j])
        self.assertEqual(first, [1, 2, 3, 4])

    def test_existing_computation_kernels_return_storage(self):
        data = cpu.storage([1, 2, 3, 4])
        for result in (cpu.broadcast(data, (2, 2), (2, 2)), cpu.matmul(data, data, (2, 2), (2, 2)),
                       cpu.transpose(data, (2, 2)), cpu.sum(data), cpu.mean(data),
                       cpu.unbroadcast(data, (2, 2), (2, 2)), cpu.softmax(data, (2, 2), 1),
                       cpu.cross_entropy(data, (2, 2), (0, 1))):
            self.assertIsInstance(result, Storage)
        normalized = cpu.layer_norm(data, cpu.full(2, 1), cpu.full(2), 2, 1e-5)
        self.assertTrue(all(isinstance(result, Storage) for result in normalized))
        self.assertIsInstance(cpu.argmax(data, (2, 2)), list)

    def test_global_norm_avoids_square_overflow(self):
        values = (cpu.storage([3e200]), cpu.storage([4e200]))
        self.assertAlmostEqual(cpu.global_norm(values) / 1e200, 5)
        self.assertEqual(cpu.global_norm(()), 0)

    def test_pooling_keeps_saved_indices_in_integer_storage(self):
        data = cpu.storage([1, 3, 4, 2])
        result, indices = cpu.max_pool2d(data, (1, 1, 2, 2), (2, 2), (2, 2), (0, 0))
        self.assertEqual(result, [4])
        self.assertIsInstance(indices, Storage)
        self.assertEqual((indices.dtype, indices.tolist()), ("int64", [2]))
        self.assertEqual(indices.buffer.typecode, "q")
        self.assertEqual(cpu.max_pool2d_backward(cpu.storage([5]), indices, 4), [0, 0, 5, 0])

    def test_complex_readouts_and_pullbacks_match_a_bell_state(self):
        state = cpu.storage([1 / math.sqrt(2), 0, 0, 1j / math.sqrt(2)], dtype="complex128")
        self.assertAlmostEqual(cpu.inner(state, state).real, 1)
        probabilities = cpu.probabilities(state, 2, (1, 0))
        for actual, expected in zip(probabilities, [0.5, 0, 0, 0.5], strict=True): self.assertAlmostEqual(actual, expected)
        gradient = cpu.storage([2, 3])
        incoming = cpu.probabilities_backward(gradient, state, 2, (0,))
        expected = [math.sqrt(2), 0, 0, 3j / math.sqrt(2)]
        for actual, value in zip(incoming, expected, strict=True): self.assertAlmostEqual(abs(actual - value), 0)
        reduced = cpu.real_inner(state, incoming, scale=2)
        self.assertAlmostEqual(reduced[0], 5)
        collapsed = cpu.collapse(state, 2, (0,), 1, probabilities[3])
        self.assertEqual(collapsed, [0, 0, 0, 1j])

    def test_weighted_sum_is_a_complex_buffer_without_mutating_inputs(self):
        first = cpu.storage([1j, 2], dtype="complex128")
        second = cpu.storage([3, -1j], dtype="complex128")
        result = cpu.weighted_sum((first, second), cpu.storage([2, -0.5]))
        self.assertEqual(result, [-1.5 + 2j, 4 + 0.5j])
        self.assertEqual(first, [1j, 2])
        self.assertEqual(cpu.basis_state(2), [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
