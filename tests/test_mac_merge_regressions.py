import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from qutai import SafeTensors, Tensor, concatenate, is_available, nn, save_safetensors
from qutai.core import operations
from qutai.backends import cpu


class MacMergeTests(unittest.TestCase):
    def test_new_operations_and_indexing_match_on_available_devices(self):
        for device in ('cpu', 'apple'):
            if not is_available(device): continue
            with self.subTest(device=device):
                x = Tensor([-2, 0, 0.5, 2], device=device, requires_grad=True)
                result = x.tanh() + x.atan()
                result.sum().backward()
                for value, actual, gradient in zip([-2, 0, 0.5, 2], result.tolist(), x.grad.tolist(), strict=True):
                    self.assertAlmostEqual(actual, math.tanh(value) + math.atan(value), places=5)
                    self.assertAlmostEqual(gradient, 1-math.tanh(value)**2+1/(1+value*value), places=5)
                x = Tensor([[1, 2, 3], [4, 5, 6]], device=device, requires_grad=True)
                pieces = (x[::-1, ::-1], x[:, :0], x[None, ..., 1])
                self.assertEqual(pieces[0].tolist(), [[6, 5, 4], [3, 2, 1]])
                self.assertEqual(pieces[1].shape, (2, 0))
                self.assertEqual(pieces[2].tolist(), [[2, 5]])
                output = concatenate((x[:, :1], x[:, 1:]), axis=-1)
                output.square().sum().backward()
                self.assertEqual(output.tolist(), x.tolist())
                self.assertEqual(x.grad.tolist(), [[2, 4, 6], [8, 10, 12]])
                self.assertEqual([part.shape for part in x.split(3)], [(2, 1)]*3)
                target = Tensor([[0]*3]*2, device=device)
                storage = target.storage
                target.copy_(x)
                self.assertIs(target.storage, storage)
                self.assertEqual(target.tolist(), x.tolist())
                drop = nn.Dropout(0.5, seed=1, device=device)
                output = drop([[1]*16])
                self.assertEqual(output.device, device)
                self.assertLessEqual(set(output.tolist()[0]), {0, 2})
                model = nn.Model([drop], device=device)
                self.assertEqual(model([[1]*16], train=False).tolist(), [[1]*16])
                with TemporaryDirectory() as directory:
                    path = Path(directory)/'dropout.json'
                    model.save(path)
                    restored = nn.Model(device=device).load(path)
                    self.assertEqual(model([[1]*16]).tolist(), restored([[1]*16]).tolist())

    def test_concatenate_accepts_raw_storage_operands(self):
        left = SimpleNamespace(storage=cpu.storage([1, 2]), shape=(1, 2), ndim=2)
        right = SimpleNamespace(storage=cpu.storage([3, 4]), shape=(1, 2), ndim=2)
        values, shape, backward = operations.concatenate((left, right), axis=0)
        self.assertEqual((values.tolist(), shape), ([1, 2, 3, 4], (2, 2)))
        self.assertEqual([part.tolist() for part in backward(cpu.storage([1, 2, 3, 4]))], [[1, 2], [3, 4]])

    @unittest.skipUnless(is_available('apple'), 'Apple GPU unavailable')
    def test_safetensors_cpu_weights_load_on_apple_and_roundtrip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'weights.safetensors'
            save_safetensors(path, {'weight': Tensor([[1, 2], [3, 4]])}, metadata={'source': 'cpu'})
            loaded = SafeTensors(path, device='apple')
            value = loaded['weight']
            self.assertEqual((value.device, value.dtype), ('apple', 'float32'))
            self.assertEqual(value.tolist(), [[1, 2], [3, 4]])
            self.assertIs(value, loaded['weight'])
            self.assertEqual(loaded.metadata, {'source': 'cpu'})
            target = Tensor([[0, 0], [0, 0]], device='apple')
            target.copy_(value)
            save_safetensors(path, {'weight': target})
            restored = SafeTensors(path)['weight']
            self.assertEqual(restored.dtype, 'float32')
            self.assertEqual(restored.tolist(), target.tolist())
