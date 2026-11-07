import ast
import unittest
from graphlib import CycleError, TopologicalSorter
from importlib.util import resolve_name
from pathlib import Path
from unittest.mock import patch

from qutai import Tensor
from qutai.backends.device import get_backend
from qutai.core import operations
from qutai.core.context import set_recording
from qutai.core.types import Shape


class Record:
    """仅提供 Operand 属性，没有计算图字段，也不继承 Tensor。"""
    def __init__(self, values: list[float], shape: Shape, requires_grad: bool = True) -> None:
        self.storage, self.shape, self.requires_grad = get_backend().storage(values), shape, requires_grad

    @property
    def size(self) -> int:
        return self.storage.size

    @property
    def ndim(self) -> int:
        return len(self.shape)


class CoreArchitectureTests(unittest.TestCase):
    def test_import_graph_is_acyclic_and_raw_kernels_stay_below_tensor(self):
        root = Path(__file__).resolve().parents[1] / 'src' / 'qutai'
        files = [root / '__init__.py', *(root / 'core').rglob('*.py'), *(root / 'backends').rglob('*.py')]
        modules = {}
        for path in files:
            parts = list(path.relative_to(root).with_suffix('').parts)
            if parts[-1] == '__init__': parts.pop()
            name = '.'.join(['qutai', *parts])
            package = name if path.name == '__init__.py' else name.rpartition('.')[0]
            modules[name] = path, package, ast.parse(path.read_text(encoding='utf-8'))

        graph = {}
        for name, (path, package, tree) in modules.items():
            dependencies = set()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)): continue
                self.assertIn(node, tree.body, f'{path.name}:{node.lineno} 的 import 必须在模块顶层')
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                else:
                    base = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                    targets = [base, *(f'{base}.{alias.name}' for alias in node.names)]
                dependencies.update(target for target in targets if target in modules)
            graph[name] = dependencies
            if name.startswith('qutai.core.operations'):
                forbidden = dependencies & {'qutai', 'qutai.core.tensor', 'qutai.core.grad', 'qutai.core.context'}
                self.assertFalse(forbidden, f'{name} 不能依赖 Tensor、包入口或计算图状态：{sorted(forbidden)}')
        try:
            list(TopologicalSorter(graph).static_order())
        except CycleError as error:
            self.fail('模块依赖成环：' + ' → '.join(error.args[1]))


class RawKernelTests(unittest.TestCase):
    def test_broadcast_kernel_accepts_protocol_only_and_returns_local_rule(self):
        left, right = Record([1.0, 2.0], (2, 1)), Record([10.0, 20.0, 30.0], (3,))
        with patch.object(Tensor, '__init__', side_effect=AssertionError('底层运算不应创建 Tensor')):
            result = operations.add(left, right)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        values, shape, rule = result
        self.assertEqual(shape, (2, 3))
        self.assertEqual(values, [11, 21, 31, 12, 22, 32])
        self.assertEqual(rule(get_backend().storage([1, 2, 3, 4, 5, 6])), ([6, 15], [5, 7, 9]))
        self.assertEqual(left.storage.tolist(), [1, 2])
        self.assertEqual(right.storage.tolist(), [10, 20, 30])
        for operand in (left, right):
            self.assertTrue(operand.requires_grad)
            self.assertEqual(set(vars(operand)), {'storage', 'shape', 'requires_grad'})

    def test_embedding_kernel_keeps_integer_indices_and_accumulates_weight_rule(self):
        weight = Record([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        with patch.object(Tensor, '__init__', side_effect=AssertionError('底层查表不应创建 Tensor')):
            values, shape, rule = operations.embedding([[2, 0], [2, 1]], weight)
        self.assertEqual(shape, (2, 2, 2))
        self.assertEqual(values, [5, 6, 1, 2, 5, 6, 3, 4])
        self.assertEqual(rule(get_backend().storage([1, 2, 3, 4, 5, 6, 7, 8])), ([3, 4, 7, 8, 6, 8],))
        self.assertEqual(set(vars(weight)), {'storage', 'shape', 'requires_grad'})

    def test_recording_is_owned_by_tensor_not_the_raw_kernel(self):
        operand = Record([2.0, 3.0], (2,))
        leaf = Tensor([2, 3], requires_grad=True)
        with set_recording(False):
            values, shape, rule = operations.square(operand)
            detached = leaf.square()
        self.assertEqual((values, shape), ([4, 9], (2,)))
        self.assertEqual(rule(get_backend().storage([3, 4])), ([12, 24],))
        self.assertTrue(operand.requires_grad)
        self.assertEqual(set(vars(operand)), {'storage', 'shape', 'requires_grad'})
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.parents, ())
        self.assertIsNone(detached.backward_rule)
        attached = leaf.square()
        self.assertTrue(attached.requires_grad)
        self.assertEqual(attached.parents, (leaf,))
        attached.backward([3, 4])
        self.assertEqual(leaf.grad.tolist(), [12, 24])


if __name__ == '__main__':
    unittest.main()
