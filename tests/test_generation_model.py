import json
import unittest
from math import isfinite
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from qutai import Stream, Tensor, generation, nn
from qutai.core.context import is_recording, is_training, set_recording, set_training


def make_model(method, seed=42):
    network = nn.DiT(1, 2, patch_size=1, features=2, heads=1, depth=1, hidden_features=3, seed=seed)
    network.output.weight.values = [0.1, -0.2]
    network.output.bias.values = [0.05]
    network.blocks[0].attention.key.weight = network.blocks[0].attention.query.weight
    network.output_norm.weight = network.blocks[0].attention_norm.weight
    network.parameters()[0].requires_grad = False
    return generation.Model(network, method=method)


class GenerationModelTests(unittest.TestCase):
    def setUp(self):
        self.images = Tensor([-1, -0.3, 0.4, 1], shape=(1, 1, 2, 2))

    def test_both_methods_prepare_pairs_and_backpropagate_finite_gradients(self):
        for method in (generation.Diffusion(2), generation.FlowMatching(2)):
            with self.subTest(method=type(method).__name__):
                model = make_model(method)
                images = (self.images + 1) / 2
                mixed, times, target = next(iter(Stream(images, method=method, seed=42, shuffle=False, progress=False)))
                direct = method.prepare(images * 2 - 1, Random(42))
                self.assertEqual((mixed.values, times, target.values), (direct[0].values, direct[1], direct[2].values))
                self.assertEqual(mixed.shape, self.images.shape)
                self.assertEqual(target.shape, self.images.shape)
                self.assertEqual(len(times), 1)
                prediction = model(mixed, times)
                self.assertEqual(prediction.shape, target.shape)
                loss = nn.MSE()(prediction, target)
                self.assertTrue(isfinite(loss.item()))
                loss.backward()
                optimizer = nn.SGD(model)
                self.assertEqual(optimizer.parameters, model.network.parameters())
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(all(isfinite(value) for value in parameter.grad.values))
                    else:
                        self.assertIsNone(parameter.grad)
                self.assertFalse(model(mixed, times, train=False).requires_grad)

    def test_sample_equals_trajectory_final_state_without_mutating_model(self):
        for method in (generation.Diffusion(2), generation.FlowMatching(2)):
            with self.subTest(method=type(method).__name__):
                model = make_model(method)
                parameters = model.parameters()
                for parameter in parameters:
                    parameter.grad = Tensor([0.2] * parameter.size, shape=parameter.shape)
                values = [parameter.values[:] for parameter in parameters]
                gradients = [parameter.grad for parameter in parameters]
                flags = [parameter.requires_grad for parameter in parameters]
                for enabled in (True, False):
                    with set_recording(enabled), set_training(enabled):
                        outside = is_recording(), is_training()
                        final = model.sample(count=2, seed=7)
                        trajectory = model.trajectory(count=2, seed=7)
                        self.assertEqual((is_recording(), is_training()), outside)
                    self.assertEqual(len(trajectory), 3)
                    self.assertEqual(final.shape, (2, 1, 2, 2))
                    self.assertEqual(final.values, trajectory[-1].values)
                    for state in (final, *trajectory):
                        self.assertFalse(state.requires_grad)
                        self.assertEqual(state.parents, ())
                self.assertEqual([parameter.values for parameter in parameters], values)
                self.assertEqual([parameter.requires_grad for parameter in parameters], flags)
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    self.assertIs(parameter.grad, gradient)

    def test_complete_and_in_place_load_preserve_method_sharing_and_references(self):
        for method_type in (generation.Diffusion, generation.FlowMatching):
            with self.subTest(method=method_type.__name__), TemporaryDirectory() as folder:
                source = make_model(method_type(2))
                target = make_model(method_type(2), seed=7)
                target.parameters()[0].requires_grad = True
                optimizer = nn.SGD(target)
                flags = [parameter.requires_grad for parameter in target.parameters()]
                network, block = target.network, target.network.blocks[0]
                for parameter in target.parameters():
                    parameter.grad = Tensor([0.3] * parameter.size, shape=parameter.shape)
                path = Path(folder) / 'model.json'
                source.save(path)
                payload = json.loads(path.read_text())
                self.assertEqual(set(payload), {'model', 'parameters', 'generation', 'value_range'})
                self.assertEqual(payload['generation'], method_type(2).config())
                self.assertEqual(payload['value_range'], [-1.0, 1.0])
                self.assertEqual(len(payload['parameters']), len(source.parameters()))
                restored = generation.Model().load(path)
                self.assertIs(type(restored.method), method_type)
                self.assertIsInstance(restored.network, nn.DiT)
                self.assertEqual(restored.method.config(), source.method.config())
                restored_block = restored.network.blocks[0]
                self.assertIs(restored_block.attention.query.weight, restored_block.attention.key.weight)
                self.assertIs(restored.network.output_norm.weight, restored_block.attention_norm.weight)
                self.assertEqual([parameter.requires_grad for parameter in restored.parameters()],
                                 [parameter.requires_grad for parameter in source.parameters()])
                self.assertEqual(restored(self.images, [0.5], train=False).values, source(self.images, [0.5], train=False).values)
                self.assertEqual(restored.sample(seed=7).values, source.sample(seed=7).values)

                self.assertIs(target.load(path), target)
                self.assertIs(target.network, network)
                self.assertIs(target.network.blocks[0], block)
                for previous, current in zip(optimizer.parameters, target.parameters(), strict=True):
                    self.assertIs(previous, current)
                    self.assertIsNone(current.grad)
                self.assertEqual([parameter.requires_grad for parameter in target.parameters()], flags)
                self.assertEqual(target.sample(seed=7).values, source.sample(seed=7).values)

    def test_wrong_method_or_late_parameter_error_does_not_partly_load(self):
        source = make_model(generation.Diffusion(2))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            source.save(path)
            original = path.read_text()
            targets = (make_model(generation.Diffusion(3), seed=7), make_model(generation.FlowMatching(2), seed=7))
            for target in targets:
                before = [parameter.values[:] for parameter in target.parameters()]
                with self.assertRaisesRegex(ValueError, '方法'):
                    target.load(path)
                self.assertEqual([parameter.values for parameter in target.parameters()], before)

            target = make_model(generation.Diffusion(2), seed=7)
            parameters = target.parameters()
            for parameter in parameters:
                parameter.grad = Tensor([0.4] * parameter.size, shape=parameter.shape)
            before = [parameter.values[:] for parameter in parameters]
            gradients = [parameter.grad for parameter in parameters]
            payload = json.loads(original)
            payload['parameters'][-1]['values'] = ['invalid'] * parameters[-1].size
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                target.load(path)
            self.assertEqual([parameter.values for parameter in parameters], before)
            for parameter, gradient in zip(parameters, gradients, strict=True):
                self.assertIs(parameter.grad, gradient)
            empty = generation.Model()
            with self.assertRaises(ValueError):
                empty.load(path)
            self.assertIsNone(empty.network)
            self.assertIsNone(empty.method)

    def test_missing_method_metadata_is_rejected(self):
        model = make_model(generation.Diffusion(2))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            model.save(path)
            payload = json.loads(path.read_text())
            del payload['generation']
            path.write_text(json.dumps(payload), encoding='utf-8')
            target = generation.Model()
            with self.assertRaisesRegex(ValueError, '方法'):
                target.load(path)
            self.assertIsNone(target.network)

    def test_unconfigured_model_default_method_and_invalid_sample_counts(self):
        configured = make_model(generation.Diffusion(2))
        self.assertIsInstance(generation.Model(configured.network).method, generation.Diffusion)
        with TemporaryDirectory() as folder:
            for model in (generation.Model(), generation.Model(method=generation.FlowMatching(2))):
                self.assertEqual(model.parameters(), ())
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    model(self.images, [0.5])
                with self.assertRaises(ValueError):
                    model.sample()
                with self.assertRaises(ValueError):
                    model.trajectory()
                with self.assertRaises(ValueError):
                    model.save(Path(folder) / 'empty.json')
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    configured.sample(count=count)
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    configured.trajectory(count=count)


if __name__ == '__main__':
    unittest.main()
