"""PQC 的实际随机门结构、权重共享与已有模型载入。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, nn, qnn


class PQCSerializationTests(unittest.TestCase):
    def test_hybrid_roundtrip_restores_actual_rotations_weights_and_freeze(self):
        inputs = [[0.2, -0.3], [0.6, 0.1]]
        for entangler in ('cnot', 'cz'):
            with self.subTest(entangler=entangler), TemporaryDirectory() as folder:
                pqc = qnn.PQC(2, depth=3, entangler=entangler, seed=42)
                pqc.weight.requires_grad = False
                source = nn.Model([nn.Linear(2, 2, seed=7), pqc, nn.Linear(2, 1, seed=11)])
                path = Path(folder) / 'hybrid.json'
                source.save(path)
                archive = json.loads(path.read_text())
                structure = archive['model']['layers'][1]
                self.assertEqual(structure['type'], 'PQC')
                self.assertEqual((structure['qubits'], structure['depth'], structure['entangler']), (2, 3, entangler))
                self.assertEqual(structure['rotations'], [list(row) for row in pqc.rotations])
                restored = nn.Model().load(path)
                layer = restored.layers[1]
                self.assertIsInstance(layer, qnn.PQC)
                self.assertEqual(layer.rotations, pqc.rotations)
                self.assertEqual(layer.weight.values, pqc.weight.values)
                self.assertFalse(layer.weight.requires_grad)
                self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)

    def test_existing_load_keeps_parameter_and_optimizer_references(self):
        source_layer, target_layer = qnn.PQC(2, depth=2, seed=42), qnn.PQC(2, depth=2, seed=42)
        target_layer.weight.values[:] = [0.1] * target_layer.weight.size
        target_layer.weight.requires_grad = False
        source, target = nn.Model([source_layer]), nn.Model([target_layer])
        optimizer = nn.SGD(target, learning_rate=0.1)
        target_layer.weight.grad = Tensor([1] * target_layer.weight.size, shape=target_layer.weight.shape)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'pqc.json'
            source.save(path)
            self.assertIs(target.load(path), target)
        self.assertIs(target.layers[0], target_layer)
        self.assertIs(target.parameters()[0], optimizer.parameters[0])
        self.assertIs(target.parameters()[0], target_layer.weight)
        self.assertIsNone(target_layer.weight.grad)
        self.assertFalse(target_layer.weight.requires_grad)
        self.assertEqual(target_layer.weight.values, source_layer.weight.values)

    def test_shared_pqc_weights_stay_shared_and_training_can_continue(self):
        layer = qnn.PQC(2, depth=1, seed=42)
        source = nn.Model([layer, layer])
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'shared.json'
            source.save(path)
            self.assertEqual(len(json.loads(path.read_text())['parameters']), 1)
            restored = nn.Model().load(path)
        self.assertIs(restored.layers[0].weight, restored.layers[1].weight)
        self.assertEqual(restored.layers[0].rotations, layer.rotations)
        inputs = [[0.2, -0.4], [0.6, 0.1]]
        self.assertEqual(restored(inputs, train=False).values, source(inputs, train=False).values)
        optimizer = nn.SGD(restored, learning_rate=0.1)
        before = restored.parameters()[0].values[:]
        restored(inputs).square().mean().backward()
        self.assertIsNotNone(restored.parameters()[0].grad)
        optimizer.step()
        self.assertNotEqual(restored.parameters()[0].values, before)

    def test_existing_load_rejects_different_axes_or_entangler_before_updating(self):
        source = nn.Model([qnn.PQC(2, depth=2, seed=42)])
        alternatives = [qnn.PQC(2, depth=2, seed=7), qnn.PQC(2, depth=2, entangler='cz', seed=42)]
        self.assertNotEqual(alternatives[0].rotations, source.layers[0].rotations)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'source.json'
            source.save(path)
            for layer in alternatives:
                with self.subTest(entangler=layer.entangler, rotations=layer.rotations):
                    target = nn.Model([layer])
                    before = layer.weight.values[:]
                    with self.assertRaises(ValueError): target.load(path)
                    self.assertEqual(layer.weight.values, before)

    def test_invalid_saved_axis_layout_and_weight_shape_are_rejected(self):
        source = nn.Model([qnn.PQC(3, depth=2, seed=42)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'pqc.json'
            source.save(path)
            original = json.loads(path.read_text())
            for broken in ('axis', 'layout', 'weight'):
                with self.subTest(broken=broken):
                    archive = json.loads(json.dumps(original))
                    if broken == 'axis': archive['model']['layers'][0]['rotations'][0][0] = 'H'
                    elif broken == 'layout': archive['model']['layers'][0]['rotations'][0].pop()
                    else: archive['parameters'][0]['shape'] = [3, 2]
                    path.write_text(json.dumps(archive))
                    with self.assertRaises(ValueError): nn.Model().load(path)


if __name__ == '__main__':
    unittest.main()
