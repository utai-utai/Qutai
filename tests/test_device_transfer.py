"""公共设备/精度接口；数值检查在 CPU 上执行，不代替实际 GPU 验证。"""

import json
import math
import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from qutai import Tensor, is_available, synchronize, nn, lm, generation, qnn, quantum
from qutai.backends import cpu
from qutai.backends.device import transfer
from qutai.core.context import set_recording
from qutai.core.grad import backward


class DeviceTransferTests(unittest.TestCase):
    def assertClose(self, actual, expected, places=5):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True): self.assertAlmostEqual(abs(a-b), 0, places=places)

    def test_precision_is_real_contiguous_storage(self):
        for dtype, size in (("float32", 4), ("float64", 8), ("complex64", 8), ("complex128", 16)):
            values = [1+2j, -3j] if dtype.startswith('complex') else [0.1, -3]
            data = cpu.storage(values, dtype=dtype)
            self.assertEqual((data.dtype, data.nbytes, data.size), (dtype, size*2, 2))
            self.assertTrue(memoryview(data.buffer).contiguous)
            clone = data.copy()
            clone[0] = 9
            self.assertNotEqual(data[0], clone[0])
            self.assertEqual(data[:1].dtype, dtype)
        self.assertNotEqual(cpu.storage([0.1], dtype='float32')[0], 0.1)

    def test_availability_and_sync_do_not_claim_gpu_support(self):
        self.assertTrue(is_available('cpu'))
        self.assertIsNone(synchronize())
        for name in ('cuda', 'mps', 'bad'):
            self.assertFalse(is_available(name))
            with self.assertRaisesRegex(ValueError, '设备'): synchronize(name)

    def test_transfer_identity_copy_and_invalid_targets(self):
        data = cpu.storage([1, 2])
        self.assertIs(transfer(data), data)
        copied = transfer(data, copy=True)
        self.assertIsNot(copied, data)
        self.assertEqual(copied.tolist(), data.tolist())
        converted = transfer(data, dtype='float32')
        self.assertEqual((converted.dtype, converted.tolist()), ('float32', [1, 2]))
        for device in ('cuda', 'mps', ''):
            with self.assertRaises(ValueError): transfer(data, device)

    def test_cast_backward_returns_gradient_to_original_precision(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        converted = x.to(dtype='float32')
        self.assertIs(x.to('cpu'), x)
        self.assertIsNot(x.to(copy=True), x)
        loss = converted.square().sum().to(dtype="float64") + x.sum()
        loss.backward()
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(x.grad.dtype, 'float64')
        self.assertEqual(x.grad.tolist(), [3, 5, 7])
        loss.backward()
        self.assertEqual(x.grad.tolist(), [6, 10, 14])

    def test_gradient_accumulation_uses_contribution_device(self):
        # 合成标签只检查反向调度，不执行或模拟 GPU 数值计算。
        parent = Tensor(0, requires_grad=True)
        output = Tensor(0, requires_grad=True)
        output.parents = (parent, parent)
        output.backward_rule = lambda _: (cpu.storage([2]), cpu.storage([3]))
        seed = cpu.storage([1])
        seed.device = "synthetic-output-device"
        self.assertEqual(backward(output, seed)[parent].tolist(), [5])

    def test_cast_respects_recording_mode_and_plain_inputs_follow_dtype(self):
        x = Tensor([1, 2], requires_grad=True)
        with set_recording(False): result = x.to(dtype='float32')
        self.assertFalse(result.requires_grad)
        self.assertEqual((result + [3, 4]).dtype, 'float32')
        with self.assertRaisesRegex(ValueError, '精度'): _ = result + x
        for dtype in ('int64', 'complex64', 'float16'):
            with self.assertRaises(ValueError): x.to(dtype=dtype)

    def test_empty_and_multidimensional_math_keep_dtype(self):
        x = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype='float32', requires_grad=True)
        results = [x+1, x/2, -x, x.square(), x.exp(), x.log(), x.mean(axis=1), x.sum(axis=0),
                   x.T, x.reshape((3, 2)), x[:, 1:], x @ [[1], [2], [3]], nn.Softmax()(x), nn.GELU()(x)]
        for value in results:
            self.assertEqual(value.dtype, 'float32')
            value.sum().backward()
        self.assertEqual(x.grad.dtype, 'float32')
        empty = Tensor([], shape=(0, 2), dtype='float32', requires_grad=True)
        self.assertEqual(empty.sum().dtype, 'float32')
        self.assertEqual((empty @ Tensor([1, 2], shape=(2, 1), dtype='float32')).shape, (0, 1))
        empty.sum().backward()
        self.assertEqual(empty.grad.dtype, 'float32')

    def test_conv_pool_normalization_gradients_match_float64(self):
        models = []
        for dtype in ('float32', 'float64'):
            model = nn.Model([nn.Conv2d(1, 2, 2, seed=2), nn.BatchNorm2d(2), nn.ReLU(),
                              nn.MaxPool2d(2), nn.Flatten(), nn.LayerNorm(2, eps=0.01), nn.Linear(2, 1, seed=3)]).to(dtype=dtype)
            x = Tensor(list(range(32)), shape=(2, 1, 4, 4), dtype=dtype, requires_grad=True)
            loss = nn.MSE()(model(x), [[1], [0]])
            loss.backward()
            self.assertEqual(loss.dtype, dtype)
            for p in model.parameters(): self.assertEqual(p.grad.dtype, dtype)
            self.assertEqual(model.layers[1].running_mean.dtype, dtype)
            models.append((model, x, loss))
        a, b = models
        self.assertAlmostEqual(a[2].item(), b[2].item(), places=4)
        for p, q in zip(a[0].parameters(), b[0].parameters(), strict=True): self.assertClose(p.grad.storage.tolist(), q.grad.storage.tolist(), places=4)
        self.assertClose(a[1].grad.storage.tolist(), b[1].grad.storage.tolist(), places=4)

    def test_model_migration_keeps_aliases_buffers_and_existing_optimizer(self):
        layer = nn.Linear(2, 2, seed=1)
        bn = nn.BatchNorm2d(2)
        model = nn.Model([nn.Model([layer, layer]), bn])
        bn.running_var = bn.running_mean  # Deliberate buffer alias.
        parameters = model.parameters()
        optimizer = nn.AdamW(model)
        layer([[1, 2]]).sum().backward()
        self.assertIs(model.to(dtype='float32'), model)
        self.assertEqual(model.parameters(), parameters)
        self.assertEqual(optimizer.parameters, parameters)
        self.assertIs(model.layers[0].layers[0], model.layers[0].layers[1])
        self.assertIs(bn.running_mean, bn.running_var)
        self.assertEqual(layer.weight.grad.dtype, 'float32')
        self.assertEqual(bn.running_mean.dtype, 'float32')
        optimizer.step()
        self.assertEqual(optimizer.averages[layer.weight].dtype, 'float32')

    def test_failed_model_migration_does_not_change_buffers_or_parameters(self):
        model = nn.Model([nn.Linear(2, 2), nn.BatchNorm2d(2)])
        before = [p.storage for p in model.parameters()]
        bn = model.layers[1]
        stats = bn.running_mean
        with self.assertRaises(ValueError): model.to('mps', dtype='float32')
        for p, value in zip(model.parameters(), before, strict=True): self.assertIs(p.storage, value)
        self.assertIs(bn.running_mean, stats)
        with self.assertRaises(ValueError): nn.Model().to('cuda')

    def test_optimizer_states_follow_parameters_and_resume(self):
        for optimizer_type, options in ((nn.SGD, {'momentum': 0.9}), (nn.AdamW, {})):
            with self.subTest(optimizer=optimizer_type.__name__), TemporaryDirectory() as folder:
                model = nn.Model([nn.Linear(2, 1, seed=1)])
                opt = optimizer_type(model, **options)
                nn.MSE()(model([[1, 2]]), [[0]]).backward()
                opt.step()
                model.to(dtype='float32')
                opt.to()
                for name in opt._buffers:
                    for value in getattr(opt, name).values(): self.assertEqual(value.dtype, 'float32')
                opt.zero_grad()
                nn.MSE()(model([[1, 2]]), [[0]]).backward()
                opt.step()
                model.save(Path(folder)/'model.json')
                opt.save(Path(folder)/'opt.json')
                restored = nn.Model().load(Path(folder)/'model.json')
                resumed = optimizer_type(restored, **options).load(Path(folder)/'opt.json')
                for target, current in ((model, opt), (restored, resumed)):
                    current.zero_grad()
                    nn.MSE()(target([[1, 2]]), [[0]]).backward()
                    current.step()
                for a, b in zip(model.parameters(), restored.parameters(), strict=True):
                    self.assertEqual(a.tolist(), b.tolist())

    def test_optimizer_step_automatically_migrates_existing_state(self):
        model = nn.Model([nn.Linear(1, 1, seed=1)])
        opt = nn.AdamW(model)
        model([[2]]).sum().backward()
        opt.step()
        model.to(dtype='float32')
        opt.zero_grad()
        model([[2]]).sum().backward()
        opt.step()
        for p in model.parameters():
            self.assertEqual(opt.steps[p], 2)
            self.assertEqual(opt.averages[p].dtype, 'float32')

    def test_saved_dtype_old_files_and_existing_target_precision(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'model.json'
            model = nn.Model([nn.BatchNorm2d(1)]).to(dtype='float32')
            model([[[[1, 3]]]])
            model.save(path)
            loaded = nn.Model().load(path)
            self.assertEqual(loaded.parameters()[0].dtype, 'float32')
            self.assertEqual(loaded.layers[0].running_var.dtype, 'float32')
            self.assertEqual(loaded([[[[2]]]], train=False).tolist(), model([[[[2]]]], train=False).tolist())
            target = nn.Model([nn.BatchNorm2d(1)])
            target.load(path)
            self.assertEqual(target.parameters()[0].dtype, 'float64')
            data = json.loads(path.read_text())
            for p in data['parameters']: p.pop('dtype')
            path.write_text(json.dumps(data))
            self.assertEqual(nn.Model().load(path).parameters()[0].dtype, 'float64')

    def test_float32_language_model_cache_training_and_save(self):
        model = lm.Model(lm.Tokenizer('abc'), context_length=4, features=4, heads=2, seed=1).to(dtype='float32')
        opt = nn.AdamW(model)
        for _ in range(2):
            opt.zero_grad()
            loss = nn.CrossEntropy()(model([[0, 1]]), [[1, 2]])
            loss.backward()
            opt.step()
        self.assertTrue(all(p.grad.dtype == 'float32' for p in model.parameters()))
        self.assertEqual(model.generate('a', length=5), model.generate('a', length=5, use_cache=False))
        with TemporaryDirectory() as folder:
            path = Path(folder)/'lm.json'
            model.save(path)
            restored = lm.Model().load(path)
            self.assertTrue(all(p.dtype == 'float32' for p in restored.parameters()))
            self.assertEqual(restored.generate('a', length=3), model.generate('a', length=3))

    def test_float32_generation_training_and_sampling(self):
        for method in (generation.Diffusion(steps=2), generation.FlowMatching(steps=2)):
            model = generation.Model(nn.DiT(1, 2, patch_size=1, features=2, heads=1, seed=1), method=method).to(dtype='float32')
            images = Tensor([0.0, 0.1, 0.2, 0.3], shape=(1, 1, 2, 2), dtype='float32')
            inputs, times, targets = method.prepare(images, Random(1))
            self.assertEqual(inputs.dtype, 'float32')
            opt = nn.AdamW(model)
            for _ in range(2):
                opt.zero_grad()
                nn.MSE()(model(inputs, times), targets).backward()
                opt.step()
            self.assertEqual(model.sample(seed=1).dtype, 'float32')
            self.assertTrue(all(p.grad.dtype == 'float32' for p in model.parameters()))

    def test_complex64_state_and_quantum_gradients(self):
        state = quantum.Circuit(2, [quantum.H(0), quantum.CNOT(0, 1)]).run().to(dtype='complex64')
        self.assertIs(state.to(), state)
        self.assertEqual(state.copy().dtype, 'complex64')
        self.assertClose(state.probabilities(), [0.5, 0, 0, 0.5])
        self.assertEqual(state.to(dtype='complex128').dtype, 'complex128')
        state.measure(seed=1)
        self.assertAlmostEqual(sum(state.probabilities()), 1, places=6)
        gradients = []
        for method in ('adjoint', 'parameter-shift'):
            angle = Tensor(0.3, dtype='float32', requires_grad=True)
            circuit = quantum.Circuit(1, [quantum.RY(0, angle)])
            value = circuit.expectation(quantum.Z(0), method=method)
            value.backward()
            self.assertEqual((value.dtype, angle.grad.dtype), ('float32', 'float32'))
            gradients.append(angle.grad.item())
            self.assertAlmostEqual(angle.grad.item(), -math.sin(0.3), places=5)
        self.assertClose(gradients[:1], gradients[1:])

    def test_float32_hybrid_network_keeps_all_leaf_gradients(self):
        model = nn.Model([nn.Linear(2, 2, seed=1), qnn.PQC(2, 1, seed=2), nn.Linear(2, 1, seed=3)]).to(dtype='float32')
        opt = nn.AdamW(model)
        for _ in range(2):
            opt.zero_grad()
            nn.MSE()(model([[0.1, 0.2]]), [[0.3]]).backward()
            opt.step()
        for p in model.parameters():
            self.assertEqual(p.grad.dtype, 'float32')
            self.assertTrue(all(math.isfinite(v) for v in p.grad.storage.tolist()))
