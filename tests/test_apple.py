"""Apple 实卡行为验证；其他平台跳过 GPU 用例，CPU 导入仍可单独检查。"""

import subprocess
import sys
import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qutai import Tensor, Stream, is_available, lm, nn, qnn, quantum, synchronize
from qutai.backends import apple, cpu
from qutai.backends.device import transfer


class AppleImportTests(unittest.TestCase):
    def test_cpu_import_does_not_load_or_compile_native_backend(self):
        script = '''
import ctypes
import ctypes.util
import subprocess
from unittest.mock import patch
with patch.object(ctypes, 'CDLL', side_effect=AssertionError('不能加载原生库')), patch.object(subprocess, 'run', side_effect=AssertionError('不能编译')):
    import qutai
    assert qutai.Tensor([1, 2]).sum().item() == 3
    assert qutai.is_available('cpu')
'''
        result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(is_available('apple'), '需要 Apple Silicon、Metal 设备与 Xcode 编译工具')
class AppleTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True): self.assertAlmostEqual(a, b, delta=2e-5 * max(1, abs(b)))

    def test_device_defaults_roundtrip_gradients_and_quantum(self):
        direct = Tensor([1, 2], device='apple')
        self.assertEqual((direct.device, direct.dtype), ('apple', 'float32'))
        self.assertIs(direct.to('apple'), direct)
        source = Tensor([1, 2], requires_grad=True)
        moved = source.to('apple')
        returned = moved.to('cpu')
        self.assertEqual((moved.dtype, returned.dtype), ('float32', 'float64'))
        self.assertEqual(moved.to('cpu', dtype='float32').dtype, 'float32')
        (returned.square().sum() + moved.sum().to('cpu')).backward()
        self.assertEqual((source.grad.device, source.grad.dtype), ('cpu', 'float64'))
        self.assertClose(source.grad.tolist(), [3, 5])
        indices = cpu.storage([2, 0, 2], dtype='int64')
        gpu_indices = transfer(indices, 'apple')
        restored_indices = transfer(gpu_indices, 'cpu')
        self.assertEqual((gpu_indices.dtype, restored_indices.dtype), ('int64', 'int64'))
        self.assertEqual(restored_indices.tolist(), [2, 0, 2])
        state = quantum.State(2, device='apple')
        self.assertEqual(state.dtype, 'complex64')
        self.assertEqual(state.to('cpu').dtype, 'complex128')
        self.assertEqual(state.to('cpu', dtype='complex64').dtype, 'complex64')
        bell = quantum.Circuit(2, [quantum.H(0), quantum.CNOT(0, 1)], device='apple').run()
        self.assertEqual(bell.dtype, 'complex64')
        self.assertClose(bell.probabilities(), [0.5, 0, 0, 0.5])

    def test_default_model_stream_and_optimizer_state_follow_device(self):
        model = nn.Model([nn.BatchNorm2d(1), nn.GlobalAvgPool2d(), nn.Linear(1, 1)])
        optimizer = nn.AdamW(model)
        parameters = model.parameters()
        nn.MSE()(model([[[[1, 2]]], [[[3, 4]]]]), [[0], [1]]).backward()
        optimizer.step()
        model.to('apple')
        optimizer.to()
        self.assertEqual(model.parameters(), parameters)
        for parameter in parameters:
            self.assertEqual((parameter.dtype, parameter.grad.dtype), ('float32', 'float32'))
            self.assertEqual((optimizer.averages[parameter].device, optimizer.averages[parameter].dtype), ('apple', 'float32'))
        self.assertEqual(model.layers[0].running_mean.dtype, 'float32')
        inputs, targets = next(iter(Stream([[[[1, 2]]], [[[3, 4]]]], [0, 1], device='apple', batch_size=2, shuffle=False, progress=False)))
        self.assertEqual((inputs.device, inputs.dtype), ('apple', 'float32'))
        optimizer.zero_grad()
        nn.MSE()(model(inputs), [[target] for target in targets]).backward()
        optimizer.step()
        model.to('cpu')
        optimizer.to()
        self.assertEqual(model.layers[0].running_var.dtype, 'float64')
        for parameter in parameters:
            self.assertEqual((parameter.dtype, parameter.grad.dtype, optimizer.averages[parameter].dtype), ('float64', 'float64', 'float64'))

    def test_storage_copy_cast_and_sync(self):
        x = Tensor([0.1, 2, 3], requires_grad=True)
        values = x.to('apple', dtype='float32')
        self.assertIn('Apple', apple.device_name())
        self.assertEqual((values.device, values.dtype, values.storage.nbytes), ('apple', 'float32', 12))
        self.assertIs(values.to('apple'), values)
        clone = values.to(copy=True)
        apple.copy_into(clone.storage, apple.full(3, 5))
        self.assertClose(values.tolist(), [0.1, 2, 3])
        self.assertEqual(clone.tolist(), [5, 5, 5])
        values.square().sum().backward()
        self.assertEqual((x.grad.device, x.grad.dtype), ('cpu', 'float64'))
        self.assertClose(x.grad.tolist(), [0.2, 4, 6])
        self.assertIsNone(synchronize('apple'))
        self.assertEqual(Tensor([], dtype='float32').to('apple').sum().item(), 0)
        self.assertEqual(Tensor([], dtype='float32').to('apple').tolist(), [])
        with self.assertRaisesRegex(ValueError, 'float32'): x.to('apple', dtype='float64')
        with self.assertRaises(AttributeError): apple.unimplemented_operation()

    def test_broadcast_reductions_transpose_and_gradients(self):
        leaves = []
        for device in ('cpu', 'apple'):
            x = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype='float32', device=device, requires_grad=True)
            bias = Tensor([0.1, -0.2, 0.3], dtype='float32', device=device, requires_grad=True)
            output = ((x + bias).T / 2).square().mean(axis=-1).sum()
            output.backward()
            leaves.append((output.item(), x.grad.storage.tolist(), bias.grad.storage.tolist()))
        self.assertAlmostEqual(leaves[0][0], leaves[1][0], places=5)
        self.assertClose(leaves[0][1], leaves[1][1])
        self.assertClose(leaves[0][2], leaves[1][2])
        source = Tensor([1, 2, 3, 4], shape=(2, 2), device='apple', dtype='float32', requires_grad=True)
        y = source.to('cpu')
        (y.square().sum().to('apple') + source.sum()).backward()
        self.assertClose(source.grad.storage.tolist(), [3, 5, 7, 9])

    def test_batched_matmul_forward_and_backward_match_cpu(self):
        results = []
        for device in ('cpu', 'apple'):
            x = Tensor(range(12), shape=(2, 2, 3), dtype='float32', device=device, requires_grad=True)
            w = Tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], shape=(3, 2), dtype='float32', device=device, requires_grad=True)
            output = x @ w
            output.square().mean().backward()
            results.append((output.storage.tolist(), x.grad.storage.tolist(), w.grad.storage.tolist()))
        for actual, expected in zip(results[1], results[0], strict=True): self.assertClose(actual, expected)
        empty = Tensor([], shape=(2, 0), device='apple', dtype='float32')
        self.assertEqual((empty @ Tensor([], shape=(0, 3), device='apple', dtype='float32')).tolist(), [[0]*3]*2)

    def test_two_step_training_and_optimizer_state_stay_on_gpu(self):
        for kind, options in ((nn.SGD, {'momentum': 0.9}), (nn.AdamW, {})):
            with self.subTest(optimizer=kind.__name__):
                models = [nn.Model([nn.Linear(2, 3, seed=42), nn.ReLU(), nn.Linear(3, 1, seed=43)]).to(device, dtype='float32')
                          for device in ('cpu', 'apple')]
                optimizers = [kind(model, learning_rate=0.01, **options) for model in models]
                initial = nn.MSE()(models[1]([[1, 2], [3, 4]], train=False), [[2], [4]]).item()
                dispatches = apple._runtime().q_dispatches()
                for _ in range(2):
                    for index, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
                        optimizer.zero_grad()
                        if index == 0:
                            loss = nn.MSE()(model([[1, 2], [3, 4]]), [[2], [4]])
                            loss.backward()
                            optimizer.step()
                        else:
                            with patch.object(apple.AppleStorage, 'tolist', side_effect=AssertionError('训练中不能导出 GPU 列表')), \
                                 patch.object(cpu, 'matmul', side_effect=AssertionError('不能回退 CPU')):
                                loss = nn.MSE()(model([[1, 2], [3, 4]]), [[2], [4]])
                                loss.backward()
                                optimizer.step()
                        for parameter in model.parameters():
                            self.assertEqual(parameter.grad.device, parameter.device)
                    for a, b in zip(models[1].parameters(), models[0].parameters(), strict=True):
                        self.assertClose(a.storage.tolist(), b.storage.tolist())
                        self.assertClose(a.grad.storage.tolist(), b.grad.storage.tolist())
                self.assertGreater(apple._runtime().q_dispatches(), dispatches)
                final = nn.MSE()(models[1]([[1, 2], [3, 4]], train=False), [[2], [4]]).item()
                self.assertLess(final, initial)
                for name in optimizers[1]._buffers:
                    for state in getattr(optimizers[1], name).values(): self.assertEqual(state.device, 'apple')

    def test_save_load_resume_and_migration_keep_parameters(self):
        with TemporaryDirectory() as folder:
            model = nn.Model([nn.Linear(2, 1, seed=42)]).to('apple', dtype='float32')
            optimizer = nn.AdamW(model)
            nn.MSE()(model([[1, 2]]), [[3]]).backward()
            optimizer.step()
            model.save(Path(folder)/'model.json')
            optimizer.save(Path(folder)/'optimizer.json')
            loaded = nn.Model().load(Path(folder)/'model.json').to('apple')
            resumed = nn.AdamW(loaded).load(Path(folder)/'optimizer.json')
            for current, update in ((model, optimizer), (loaded, resumed)):
                update.zero_grad()
                nn.MSE()(current([[1, 2]]), [[3]]).backward()
                update.step()
            for a, b in zip(model.parameters(), loaded.parameters(), strict=True): self.assertClose(a.storage.tolist(), b.storage.tolist())
            parameters = model.parameters()
            model.to('cpu', dtype='float64')
            optimizer.to()
            self.assertEqual(model.parameters(), parameters)
            for value in optimizer.averages.values(): self.assertEqual((value.device, value.dtype), ('cpu', 'float64'))
            self.assertFalse(is_available('mps'))

    def test_transformer_classification_and_repeated_embedding_gradients(self):
        results = []
        for device in ('cpu', 'apple'):
            embedding = nn.Embedding(5, 4, seed=42).to(device, dtype='float32')
            model = nn.Model([nn.TransformerBlock(4, heads=2, hidden_features=8, seed=43), nn.Linear(4, 3, seed=44)]).to(device, dtype='float32')
            parameters = embedding.parameters() + model.parameters()
            output = model(embedding([[1, 1, 3], [2, 4, 2]]))
            loss = nn.CrossEntropy()(output, [[0, 2, 1], [1, 0, 2]])
            loss.backward()
            norm = nn.clip_grad_norm(parameters, max_norm=0.7)
            results.append((output.storage.tolist(), [p.grad.storage.tolist() for p in parameters], norm))
            self.assertEqual(nn.Softmax(axis=1)(output).device, device)
            self.assertEqual(len(output.argmax(axis=-1)), 2)
        self.assertClose(results[1][0], results[0][0])
        for actual, expected in zip(results[1][1], results[0][1], strict=True): self.assertClose(actual, expected)
        self.assertAlmostEqual(results[1][2], results[0][2], places=4)
        tokenizer = lm.Tokenizer('abcabc')
        model = lm.Model(tokenizer, context_length=4, features=4, heads=2, hidden_features=8, seed=42).to('apple', dtype='float32')
        self.assertEqual(model.generate('ab', length=7), model.generate('ab', length=7, use_cache=False))
        x = Tensor([-1, 0.1, 2], device='apple', dtype='float32', requires_grad=True)
        x.exp().log().sum().backward()
        self.assertClose(x.grad.tolist(), [1, 1, 1])

    def test_convolution_batchnorm_pool_training_and_eval_match_cpu(self):
        results = []
        for device in ('cpu', 'apple'):
            model = nn.Model([nn.Conv2d(2, 3, (2, 3), stride=(1, 2), padding=(1, 1), bias=False, seed=42), nn.BatchNorm2d(3), nn.ReLU(), nn.MaxPool2d(2, stride=1), nn.Flatten(), nn.Linear(24, 2, seed=43)]).to(device, dtype='float32')
            samples = Tensor(Random(42).choices([i / 101 for i in range(-101, 101)], k=2*2*4*5), shape=(2, 2, 4, 5), requires_grad=True, device=device, dtype='float32')
            optimizer = nn.AdamW(model)
            for _ in range(2):
                optimizer.zero_grad()
                output = model(samples)
                loss = nn.CrossEntropy()(output, [0, 1])
                loss.backward()
                optimizer.step()
            prediction = model(samples, train=False)
            results.append((prediction.storage.tolist(), samples.grad.storage.tolist(), [p.grad.storage.tolist() for p in model.parameters()], [v.tolist() for v in model.layers[1].buffers().values()]))
        for field in (0, 1): self.assertClose(results[1][field], results[0][field])
        for field in (2, 3):
            for actual, expected in zip(results[1][field], results[0][field], strict=True): self.assertClose(actual, expected)
        images, labels = next(iter(Stream([[[[1, 2], [3, 4]]]]*2, [0, 1], batch_size=2, shuffle=False, device='apple', dtype='float32', progress=False)))
        self.assertEqual((images.device, images.dtype, images.shape, labels), ('apple', 'float32', (2, 1, 2, 2), [0, 1]))

    def test_large_and_odd_matmul_and_async_lifetimes(self):
        for rows, inner, columns in ((32, 16, 16), (17, 19, 23)):
            results = []
            for device in ('cpu', 'apple'):
                left = Tensor([(i%11-5)/11 for i in range(rows*inner)], shape=(rows, inner), dtype='float32', device=device, requires_grad=True)
                right = Tensor([(i%7-3)/7 for i in range(inner*columns)], shape=(inner, columns), dtype='float32', device=device, requires_grad=True)
                output = left @ right
                output.square().mean().backward()
                results.append((output.storage.tolist(), left.grad.storage.tolist(), right.grad.storage.tolist()))
            for actual, expected in zip(results[1], results[0], strict=True): self.assertClose(actual, expected)
        value = Tensor([1, 2, 3], device='apple', dtype='float32')
        for _ in range(150): value = value * 0.99 + 0.01
        self.assertClose(value.tolist(), [1, 1+0.99**150, 1+2*0.99**150])

    def test_quantum_state_gradients_and_hybrid_serialization(self):
        results = []
        for device in ('cpu', 'apple'):
            angle = Tensor(0.7, device=device, dtype='float32', requires_grad=True)
            circuit = quantum.Circuit(3, [quantum.H(0), quantum.RX(1, angle), quantum.CNOT(0, 2), quantum.CZ(1, 2), quantum.RY(2, angle)])
            gradients = []
            for method in ('adjoint', 'parameter-shift'):
                angle.zero_grad()
                loss = circuit.probabilities(wires=(2, 0), method=method).square().sum() + circuit.expectation(quantum.X(1) + quantum.Z(2), method=method)
                loss.backward()
                gradients.append(angle.grad.item())
            state = circuit.run()
            results.append((state.storage.tolist(), state.probabilities((2, 0)), gradients, state.expectation(quantum.Y(1))))
            state.measure(wires=0, seed=42)
            self.assertAlmostEqual(sum(state.probabilities()), 1, places=6)
            self.assertAlmostEqual(gradients[0], gradients[1], places=5)
        for field in (0, 1, 2): self.assertClose(results[1][field], results[0][field])
        self.assertAlmostEqual(results[1][3], results[0][3], places=5)
        bell = quantum.Circuit(2, [quantum.H(0), quantum.CNOT(0, 1)], device='apple', dtype='complex64').run()
        self.assertEqual(bell.device, 'apple')
        self.assertClose(bell.probabilities(), [0.5, 0, 0, 0.5])
        with TemporaryDirectory() as folder:
            model = nn.Model([nn.Linear(2, 2, seed=42), qnn.PQC(2, 1, seed=43)]).to('apple', dtype='float32')
            nn.MSE()(model([[0.1, 0.2]]), [[0, 1]]).backward()
            nn.SGD(model).step()
            model.save(Path(folder)/'hybrid.json')
            loaded = nn.Model().load(Path(folder)/'hybrid.json').to('apple')
            self.assertClose(loaded([[0.1, 0.2]], train=False).storage.tolist(), model([[0.1, 0.2]], train=False).storage.tolist())

    def test_accumulation_released_graph_and_matmul_backward(self):
        results = []
        for device in ('cpu', 'apple'):
            x = Tensor([0.2, -0.4, 0.7, 0.1], shape=(2, 2), device=device, dtype='float32', requires_grad=True)
            weight = Tensor([0.3, -0.2, 0.4, 0.8], shape=(2, 2), device=device, dtype='float32', requires_grad=True)
            shared = x @ weight
            loss = (shared.square() + shared).sum() + x.square().sum()
            loss.backward()
            first = [p.grad.storage.tolist() for p in (x, weight)]
            loss.backward(retain_graph=False)
            for parameter, previous in zip((x, weight), first, strict=True):
                self.assertClose(parameter.grad.storage.tolist(), [2 * value for value in previous])
            self.assertEqual(loss.parents, ())
            with self.assertRaisesRegex(RuntimeError, '计算图已释放'): loss.backward()
            results.append([p.grad.storage.tolist() for p in (x, weight)])
        for actual, expected in zip(results[1], results[0], strict=True): self.assertClose(actual, expected)

    def test_adamw_groups_and_inplace_gradient_clipping(self):
        results = []
        for device in ('cpu', 'apple'):
            x = Tensor([0.2, -0.4], device=device, dtype='float32', requires_grad=True)
            y = Tensor([0.8, 0.1], device=device, dtype='float32', requires_grad=True)
            optimizer = nn.AdamW([{'params': [x], 'learning_rate': 0.03}, {'params': [y], 'learning_rate': 0.007, 'weight_decay': 0}], learning_rate=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                ((x + y).square().sum() + x.square().sum()).backward(retain_graph=False)
                storages = (x.grad.storage, y.grad.storage)
                norm = nn.clip_grad_norm(optimizer.parameters, max_norm=0.4)
                self.assertGreater(norm, 0.4)
                self.assertIs(x.grad.storage, storages[0])
                self.assertIs(y.grad.storage, storages[1])
                if device == 'apple':
                    with patch.object(apple.AppleStorage, 'tolist', side_effect=AssertionError('AdamW 更新不能导出 GPU 数组')):
                        optimizer.step()
                else: optimizer.step()
            results.append([p.storage.tolist() for p in (x, y)])
        for actual, expected in zip(results[1], results[0], strict=True): self.assertClose(actual, expected)

    def test_device_rotation_gradients_without_host_export(self):
        results = []
        for device in ('cpu', 'apple'):
            angles = Tensor([0.3, -0.7, 1.1], device=device, dtype='float32', requires_grad=True)
            circuit = quantum.Circuit(3, [quantum.H(1), quantum.RX(0, angles[0]), quantum.RY(1, angles[1]), quantum.CNOT(0, 1), quantum.H(2), quantum.RZ(2, angles[2]), quantum.CNOT(1, 2), quantum.H(2)])
            methods = []
            for method in ('adjoint', 'parameter-shift'):
                angles.zero_grad()
                if device == 'apple':
                    with patch.object(apple.AppleStorage, 'tolist', side_effect=AssertionError('量子前向反向不能导出 GPU 数组')), \
                         patch.object(cpu, 'apply_gate', side_effect=AssertionError('不能回退 CPU 量子门')):
                        loss = circuit.probabilities(wires=(2, 0), method=method).square().sum() + circuit.expectation(quantum.X(1) + quantum.Z(2), method=method)
                        loss.backward()
                else:
                    loss = circuit.probabilities(wires=(2, 0), method=method).square().sum() + circuit.expectation(quantum.X(1) + quantum.Z(2), method=method)
                    loss.backward()
                methods.append((loss.item(), angles.grad.storage.tolist()))
            self.assertClose(methods[0][1], methods[1][1])
            results.append(methods)
        for actual, expected in zip(results[1], results[0], strict=True):
            self.assertAlmostEqual(actual[0], expected[0], places=5)
            self.assertClose(actual[1], expected[1])

    def test_decoder_dense_moe_gradients_and_chunked_checkpoint_resume(self):
        inputs = [[1, 2, 3, 1], [4, 3, 2, 4]]
        targets = [[1, 2, -100, 1], [4, 3, 2, 4]]
        for kind in ('dense', 'moe'):
            with self.subTest(kind=kind):
                config = lm.Config(vocab_size=16, n_layers=1, d_model=8, n_heads=2, n_kv_heads=1, d_ff=16, max_seq_len=8, ffn_type=kind, n_experts=4, top_k=2, d_expert=8, activation_checkpointing=True)
                models = [lm.Decoder(config, device=device, dtype='float32', seed=42) for device in ('cpu', 'apple')]
                results = []
                for model in models:
                    loss = model.loss(inputs, targets, chunk_size=3)
                    loss.backward(retain_graph=False)
                    results.append((loss.item(), [None if p.grad is None else p.grad.storage.tolist() for p in model.parameters()]))
                self.assertAlmostEqual(results[0][0], results[1][0], places=5)
                for actual, expected in zip(results[1][1], results[0][1], strict=True):
                    if expected is None: self.assertIsNone(actual)
                    else: self.assertClose(actual, expected)
                model = models[1]
                optimizer = nn.AdamW([{'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': 0.1}, {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0}], learning_rate=0.003)
                optimizer.step()
                with TemporaryDirectory() as folder:
                    lm.save_checkpoint(folder, model, optimizer, global_step=1, chunk_bytes=257, shard_bytes=2048)
                    restored = lm.Decoder(config, device='apple', dtype='float32', seed=99)
                    resumed = nn.AdamW([{'params': [p for p in restored.parameters() if p.ndim >= 2], 'weight_decay': 0.1}, {'params': [p for p in restored.parameters() if p.ndim < 2], 'weight_decay': 0}], learning_rate=0.003)
                    lm.load_checkpoint(folder, restored, resumed, chunk_bytes=113)
                    for current, update in ((model, optimizer), (restored, resumed)):
                        update.zero_grad()
                        current.loss(inputs, targets, chunk_size=3).backward(retain_graph=False)
                        update.step()
                    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
                        self.assertEqual(actual.storage.tolist(), expected.storage.tolist())
                        self.assertEqual(resumed.steps.get(actual), optimizer.steps.get(expected))
                        if actual in resumed.averages:
                            self.assertEqual(resumed.averages[actual].tolist(), optimizer.averages[expected].tolist())
                            self.assertEqual(resumed.square_averages[actual].tolist(), optimizer.square_averages[expected].tolist())
