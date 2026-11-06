"""Tensor 的自然语法、基本索引和共享参数保持同一套自动求导行为。"""

import subprocess
import sys
import unittest
from pathlib import Path

import qutai as qt
from qutai import Tensor, nn


class TensorInterfaceTests(unittest.TestCase):
    def test_arithmetic_accepts_scalars_and_lists_on_either_side(self):
        tensor = Tensor([2, 4])
        for result, expected in [
            (tensor + 2.0, [4, 6]), (2.0 + tensor, [4, 6]),
            (tensor - 2.0, [0, 2]), (2.0 - tensor, [0, -2]),
            (tensor * 2.0, [4, 8]), (2.0 * tensor, [4, 8]),
            (tensor / 2.0, [1, 2]), (2.0 / tensor, [1, 0.5]),
            (tensor + [10, 20], [12, 24]), ([10, 20] + tensor, [12, 24]),
            (tensor - [10, 20], [-8, -16]), ([10, 20] - tensor, [8, 16]),
            (tensor * [10, 20], [20, 80]), ([10, 20] * tensor, [20, 80]),
            (tensor / [1, 2], [2, 2]), ([8, 8] / tensor, [4, 2]),
            (-tensor, [-2, -4]),
        ]:
            with self.subTest(expected=expected):
                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.tolist(), expected)
        self.assertEqual(tensor.tolist(), [2, 4])

    def test_reflected_arithmetic_backward(self):
        tensor = Tensor([1, 2, 4], requires_grad=True)
        (3.0 / tensor + [2, 4, 6] - 0.5 * tensor).sum().backward()
        self.assertEqual(tensor.grad.tolist(), [-3.5, -1.25, -0.6875])

    def test_matmul_works_with_lists_on_either_side(self):
        vector = Tensor([1, 2], requires_grad=True)
        self.assertEqual((vector @ [[3, 4], [5, 6]]).tolist(), [13, 16])
        self.assertEqual(([[3, 4], [5, 6]] @ vector).tolist(), [11, 17])
        (vector @ [3, 4] + [5, 6] @ vector).backward()
        self.assertEqual(vector.grad.tolist(), [8, 10])

    def test_methods_and_math_entry_points_are_compatible(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(len(tensor), 2)
        self.assertEqual(tensor.size, 6)
        self.assertEqual(tensor.sum(axis=1, keepdims=True).tolist(), [[6], [15]])
        self.assertEqual(tensor.mean(axis=0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(tensor.reshape((3, 2)).tolist(), qt.reshape(tensor, (3, 2)).tolist())
        self.assertEqual(tensor.transpose((1, 0)).tolist(), qt.transpose(tensor, (1, 0)).tolist())
        self.assertEqual(tensor.T.tolist(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(tensor.square().tolist(), qt.square(tensor).tolist())
        self.assertEqual(tensor.argmax(axis=1), [2, 2])
        self.assertEqual(tensor.argmax(axis=0, keepdims=True), [[1, 1, 1]])
        self.assertIsInstance(tensor.argmax(), int)
        for actual, expected in zip(tensor.exp().log().values, tensor.values, strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(qt.add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(qt.mean([1, 2, 3]), 2)
        self.assertEqual(qt.add(tensor, 2).tolist(), (tensor + 2).tolist())

    def test_methods_preserve_gradients_and_T_reverses_all_axes(self):
        tensor = Tensor(list(range(12)), shape=(2, 2, 3), requires_grad=True)
        self.assertEqual(tensor.T.shape, (3, 2, 2))
        self.assertEqual(tensor.transpose((0, 2, 1)).shape, (2, 3, 2))
        tensor.T.reshape((3, 4)).square().sum().backward()
        self.assertEqual(tensor.grad.values, [2 * value for value in tensor.values])

    def test_integer_slice_tuple_ellipsis_and_new_axes(self):
        tensor = Tensor(list(range(24)), shape=(2, 3, 4))
        self.assertEqual(tensor[-1].tolist(), [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        self.assertEqual(tensor[1, ..., -2].tolist(), [14, 18, 22])
        self.assertEqual(tensor[..., 1, None].shape, (2, 3, 1))
        self.assertEqual(tensor[..., 1, None].tolist(), [[[1], [5], [9]], [[13], [17], [21]]])
        self.assertEqual(tensor[None, 1, ::-1, 1::2].tolist(), [[[21, 23], [17, 19], [13, 15]]])
        self.assertEqual(tensor[1, 2, -1].shape, ())
        self.assertEqual(tensor[1, 2, -1].item(), 23)
        self.assertEqual(tensor[()].tolist(), tensor.tolist())

    def test_reverse_slice_backward_keeps_selected_order(self):
        tensor = Tensor(list(range(12)), shape=(3, 4), requires_grad=True)
        result = tensor[::-1, 1::2]
        self.assertEqual(result.tolist(), [[9, 11], [5, 7], [1, 3]])
        result.backward([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(tensor.grad.tolist(), [[0, 5, 0, 6], [0, 3, 0, 4], [0, 1, 0, 2]])

    def test_overlapping_slices_and_reused_result_accumulate_gradients(self):
        tensor = Tensor([1, 2, 3, 4], requires_grad=True)
        (tensor[:3].sum() + 2 * tensor[1:].sum() + tensor[-1]).backward()
        self.assertEqual(tensor.grad.tolist(), [1, 3, 3, 3])
        tensor.zero_grad()
        selected = tensor[::2]
        (selected + selected).sum().backward()
        self.assertEqual(tensor.grad.tolist(), [2, 0, 2, 0])

    def test_mixed_slice_finite_differences(self):
        tensor = Tensor([index / 10 for index in range(36)], shape=(3, 3, 4), requires_grad=True)
        tensor[1:, -1::-2, None, ...].square().mean().backward()
        epsilon = 1e-6
        for index, analytic in enumerate(tensor.grad.values):
            original = tensor.values[index]
            tensor.values[index] = original + epsilon
            upper = tensor[1:, -1::-2, None, ...].square().mean().item()
            tensor.values[index] = original - epsilon
            lower = tensor[1:, -1::-2, None, ...].square().mean().item()
            tensor.values[index] = original
            self.assertAlmostEqual(analytic, (upper - lower) / (2 * epsilon), places=7)

    def test_scalar_indexing_and_length(self):
        scalar = Tensor(5, requires_grad=True)
        self.assertEqual(scalar.ndim, 0)
        self.assertEqual(scalar[()].item(), 5)
        self.assertEqual(scalar[...].shape, ())
        self.assertEqual(scalar[None, ..., None].tolist(), [[5]])
        (scalar[None, ...].sum() + scalar[()]).backward()
        self.assertEqual(scalar.grad.item(), 2)
        with self.assertRaises(TypeError):
            len(scalar)
        with self.assertRaises(IndexError):
            scalar[0]

    def test_empty_slices_and_empty_axes_backward(self):
        tensor = Tensor([1, 2, 3], requires_grad=True)
        self.assertEqual(tensor[1:1].shape, (0,))
        tensor[1:1].sum().backward()
        self.assertEqual(tensor.grad.tolist(), [0, 0, 0])
        empty = Tensor([], shape=(2, 0, 3), requires_grad=True)
        result = empty[::-1, None, :, ::-1]
        self.assertEqual(result.shape, (2, 1, 0, 3))
        self.assertEqual(result.values, [])
        result.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.values, [])
        with self.assertRaises(IndexError):
            empty[:, 0]

    def test_invalid_basic_indices_raise(self):
        tensor = Tensor([[1, 2], [3, 4]])
        for index in (2, -3, (0, 0, 0), (Ellipsis, Ellipsis)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                tensor[index]
        for index in (1.5, "1", [0, 1], ([0], slice(None))):
            with self.subTest(index=index), self.assertRaises(TypeError):
                tensor[index]
        with self.assertRaises(ValueError):
            tensor[::0]

    def test_tensors_still_use_identity_for_graphs_and_parameter_deduplication(self):
        first, second = Tensor([1, 2], requires_grad=True), Tensor([1, 2], requires_grad=True)
        self.assertEqual(len({first: 1, second: 2}), 2)
        (first * first + second).sum().backward()
        self.assertEqual(first.grad.tolist(), [2, 4])
        self.assertEqual(second.grad.tolist(), [1, 1])
        left, right = nn.Linear(2, 2, seed=1), nn.Linear(2, 2, seed=2)
        right.weight = left.weight
        model = nn.Model([left, right])
        parameters = model.parameters()
        self.assertEqual(len(parameters), 3)
        for actual, expected in zip(parameters, (left.weight, left.bias, right.bias), strict=True):
            self.assertIs(actual, expected)
        optimizer = nn.SGD(model, learning_rate=0.1)
        model([[1, 2]]).sum().backward()
        before, gradient = left.weight.values[:], left.weight.grad.values[:]
        optimizer.step()
        self.assertEqual(left.weight.values, [value - 0.1 * g for value, g in zip(before, gradient, strict=True)])
        self.assertIs(right.weight, left.weight)

    def test_tensor_submodule_works_without_package_bootstrap_binding(self):
        source = Path(__file__).resolve().parents[1] / "src" / "qutai"
        script = """
import sys
from types import ModuleType

package = ModuleType("qutai")
package.__path__ = [sys.argv[1]]
sys.modules["qutai"] = package
from qutai.core.tensor import Tensor, matmul

assert (Tensor([1, 2]) + 3).tolist() == [4, 5]
assert matmul([1, 2], [3, 4]) == 11
assert "operations" not in Tensor.__dict__
"""
        result = subprocess.run([sys.executable, "-c", script, str(source)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
