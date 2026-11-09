import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, generation, is_available, lm, nn, qnn, quantum


class DeviceConstructorTests(unittest.TestCase):
    def test_layers_accept_device_and_precision_at_creation(self):
        for device in (name for name in ('cpu', 'apple', 'cuda') if is_available(name)):
            layers = [
                nn.Embedding(4, 2, device=device, dtype='float32'),
                nn.Linear(2, 2, device=device, dtype='float32'),
                nn.Conv2d(1, 2, 1, device=device, dtype='float32'),
                nn.BatchNorm2d(2, device=device, dtype='float32'),
                nn.LayerNorm(2, device=device, dtype='float32'),
                nn.Attention(2, device=device, dtype='float32'),
                nn.MultiHeadAttention(4, 2, device=device, dtype='float32'),
                nn.FeedForward(2, device=device, dtype='float32'),
                nn.TransformerBlock(4, 2, device=device, dtype='float32'),
                nn.PatchEmbedding(1, 2, 1, device=device, dtype='float32'),
                nn.PositionEmbedding(2, 2, device=device, dtype='float32'),
                nn.ResNetBlock(1, 2, device=device, dtype='float32'),
                nn.DiT(1, 2, patch_size=1, features=2, device=device, dtype='float32'),
            ]
            for layer in layers:
                with self.subTest(device=device, layer=type(layer).__name__):
                    self.assertTrue(layer.parameters())
                    self.assertEqual({(p.device, p.dtype) for p in layer.parameters()}, {(device, 'float32')})
            bn = layers[3]
            self.assertEqual((bn.running_mean.dtype, bn.running_var.dtype), ('float32', 'float32'))
            attention = layers[6]([[[1, 2, 3, 4], [2, 1, 0, 1]]])
            attention.sum().backward()
            self.assertEqual(attention.dtype, 'float32')
            self.assertTrue(all(p.grad is not None for p in layers[6].parameters()))

    def test_parameterless_layers_place_plain_lists(self):
        cases = [
            (nn.Identity, {}, [[1, -2]]), (nn.ReLU, {}, [[1, -2]]),
            (nn.GELU, {}, [[1, -2]]), (nn.Softmax, {}, [[1, -2]]),
            (nn.Mean, {}, [[1, 2]]), (nn.Flatten, {}, [[[1, 2]]]),
            (nn.MaxPool2d, {'kernel_size': 2}, [[[[1, 2], [3, 4]]]]),
            (nn.GlobalAvgPool2d, {}, [[[[1, 2], [3, 4]]]]),
        ]
        for kind, options, data in cases:
            with self.subTest(layer=kind.__name__):
                result = kind(**options, device='cpu', dtype='float32')(data)
                self.assertEqual((result.device, result.dtype), ('cpu', 'float32'))

    def test_model_construction_preserves_shared_parameters_and_buffers(self):
        shared = nn.Linear(2, 2, seed=1)
        shared([[1, 2]]).sum().backward()
        weight, gradient = shared.weight, shared.weight.grad
        bn = nn.BatchNorm2d(2)
        bn.running_var = bn.running_mean
        model = nn.Model([nn.Model([shared, shared]), bn], device='cpu', dtype='float32')
        self.assertIs(model.layers[0].layers[0], model.layers[0].layers[1])
        self.assertIs(shared.weight, weight)
        self.assertIs(shared.weight.grad, gradient)
        self.assertEqual(gradient.dtype, 'float32')
        self.assertIs(bn.running_mean, bn.running_var)
        self.assertEqual(bn.running_mean.dtype, 'float32')

    def test_empty_models_apply_creation_precision_when_loading(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            nn.Model([nn.Linear(1, 1, seed=1)]).save(path)
            restored = nn.Model(device='cpu', dtype='float32').load(path)
            result = restored([[2]])
            result.sum().backward()
            self.assertEqual(result.dtype, 'float32')
            self.assertTrue(all(p.grad.dtype == 'float32' for p in restored.parameters()))
            nn.Model([nn.ReLU()]).save(path)
            plain = nn.Model(device='cpu', dtype='float32').load(path)
            self.assertEqual(plain([[-1, 2]]).tolist(), [[0, 2]])
            self.assertEqual(plain([[-1, 2]]).dtype, 'float32')

    def test_models_train_and_restore_with_creation_device(self):
        tokenizer = lm.Tokenizer('abc')
        language = lm.Model(tokenizer, context_length=2, features=2, heads=1, device='cpu', dtype='float32')
        generated = generation.Model(nn.DiT(1, 2, patch_size=1, features=2, heads=1),
                                     method=generation.FlowMatching(steps=2), device='cpu', dtype='float32')
        hybrid = nn.Model([nn.Linear(2, 2), qnn.PQC(2, 1), nn.Linear(2, 1)], device='cpu', dtype='float32')
        qaoa = quantum.QAOA(2, quantum.maxcut([(0, 1)]), device='cpu', dtype='float32')
        pqc = qnn.PQC(2, 1, device='cpu', dtype='float32')
        for model, loss in [
            (language, nn.CrossEntropy()(language([[0, 1]]), [[1, 2]])),
            (generated, generated([[[[0, 0], [0, 0]]]], [0.5]).square().mean()),
            (hybrid, hybrid([[0.1, 0.2]]).sum()),
            (qaoa, qaoa.expectation()), (pqc, pqc([[0.1, 0.2]]).sum()),
        ]:
            with self.subTest(model=type(model).__name__):
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.dtype == 'float32' for p in model.parameters()))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            for model, factory in [(language, lm.Model), (generated, generation.Model)]:
                model.save(path)
                restored = factory(device='cpu', dtype='float32').load(path)
                self.assertEqual({p.dtype for p in restored.parameters()}, {'float32'})
            self.assertEqual(restored.sample(count=1).dtype, 'float32')

    @unittest.skipUnless(is_available('apple'), 'Apple GPU unavailable')
    def test_apple_training_and_empty_load_follow_device_defaults(self):
        model = nn.Model([nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)], device='apple')
        optimizer = nn.AdamW(model)
        for _ in range(2):
            optimizer.zero_grad()
            loss = nn.MSE()(model([[1, 2], [2, 1]]), [[1], [0]])
            loss.backward()
            optimizer.step()
        self.assertEqual({(p.device, p.dtype) for p in model.parameters()}, {('apple', 'float32')})
        self.assertTrue(all(p.grad.device == 'apple' for p in model.parameters()))
        self.assertEqual(nn.ReLU(device='apple')([1, -2]).device, 'apple')
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            nn.Model([nn.Linear(2, 1)]).save(path)
            restored = nn.Model(device='apple').load(path)
            restored([[1, 2]]).sum().backward()
            self.assertEqual({(p.device, p.dtype) for p in restored.parameters()}, {('apple', 'float32')})

    @unittest.skipUnless(is_available('apple'), 'Apple GPU unavailable')
    def test_apple_language_generation_and_quantum_creation(self):
        language = lm.Model(lm.Tokenizer('abc'), context_length=2, features=2, heads=1, device='apple')
        generated = generation.Model(nn.DiT(1, 2, patch_size=1, features=2, heads=1),
                                     method=generation.FlowMatching(steps=2), device='apple')
        pqc = qnn.PQC(2, 1, device='apple')
        qaoa = quantum.QAOA(2, quantum.maxcut([(0, 1)]), device='apple')
        results = [language([[0, 1]]).sum(), generated([[[[0, 0], [0, 0]]]], [0.5]).sum(),
                   pqc([[0.1, 0.2]]).sum(), qaoa.expectation()]
        for model, result in zip((language, generated, pqc, qaoa), results, strict=True):
            with self.subTest(model=type(model).__name__):
                result.backward()
                self.assertEqual((result.device, result.dtype), ('apple', 'float32'))
                self.assertTrue(all(p.grad is not None and p.grad.device == 'apple' for p in model.parameters()))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            for model, factory in ((language, lm.Model), (generated, generation.Model)):
                model.save(path)
                restored = factory(device='apple').load(path)
                self.assertEqual({(p.device, p.dtype) for p in restored.parameters()}, {('apple', 'float32')})
