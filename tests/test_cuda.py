"""真实 CUDA 数值、反向和优化器对照；无 CUDA 的 CI 跳过实卡用例。"""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qutai import Tensor, nn, is_available, synchronize
from qutai.backends import cpu, cuda, cuda_runtime


class CUDAImportTests(unittest.TestCase):
    def test_cpu_import_does_not_load_cuda(self):
        code = "from qutai import Tensor; from qutai.backends import cuda_runtime; assert cuda_runtime._runtime is None; assert (Tensor([1,2])+1).tolist()==[2,3]; assert cuda_runtime._runtime is None"
        subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code], check=True)


@unittest.skipUnless(is_available("cuda"), "需要实际 CUDA 设备和 NVRTC")
class CUDATests(unittest.TestCase):
    def assertClose(self, a, b, tol=2e-5):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True): self.assertLessEqual(abs(x-y), tol * max(1, abs(y)))

    def test_storage_copy_conversion_and_empty(self):
        for dtype in ("float32", "float64", "complex64", "complex128", "int64"):
            data = [1+2j, -3j] if dtype.startswith("complex") else [1, -3]
            gpu = cuda.storage(data, dtype=dtype)
            self.assertEqual(gpu.tolist(), data)
            self.assertEqual(gpu.nbytes, len(data) * cuda._TYPES[dtype][1])
            copy = gpu.copy()
            self.assertNotEqual(copy.ptr, gpu.ptr)
            self.assertEqual(copy.tolist(), data)
            self.assertEqual(cuda.full(0, dtype=dtype).tolist(), [])
        a = cuda.storage([0.1, 2], dtype="float64")
        b = cuda.from_storage(a, dtype="float32")
        self.assertClose(b.tolist(), a.tolist())
        self.assertEqual(cuda.from_storage(b, dtype="float64").tolist(), b.tolist())
        self.assertEqual(Tensor([], shape=(0, 2), device="cuda", dtype="float32").sum().item(), 0)
        synchronize("cuda")

    def test_broadcast_matmul_reductions_transpose_and_gradients(self):
        for dtype in ("float32", "float64"):
            results = []
            for device in ("cpu", "cuda"):
                a = Tensor(list(range(12)), shape=(2, 2, 3), requires_grad=True, device=device, dtype=dtype)
                b = Tensor([.2, -.3, .4, .5, -.6, .7], shape=(1, 3, 2), requires_grad=True, device=device, dtype=dtype)
                output = ((a @ b).transpose((1, 0, 2)) + [1, 2]).square().mean(axis=1)
                output.sum().backward()
                results.append([output.storage.tolist(), a.grad.storage.tolist(), b.grad.storage.tolist()])
            for a, b in zip(*results): self.assertClose(a, b)

    def test_selection_embedding_stack_and_repeated_scatter(self):
        results = []
        for device in ("cpu", "cuda"):
            layer = nn.Embedding(4, 3, seed=2).to(device, dtype="float32")
            output = layer([[1, 1, 3], [0, 2, 1]])
            output[:, 1:].sum().backward()
            results.append([output.storage.tolist(), layer.weight.grad.storage.tolist()])
        for a, b in zip(*results): self.assertClose(a, b)

    def test_device_cast_backward_returns_to_original(self):
        original = Tensor([1., 2., 3.], requires_grad=True)
        target = original.to("cuda", dtype="float32")
        target.square().sum().backward()
        self.assertEqual(original.grad.device, "cpu")
        self.assertEqual(original.grad.dtype, "float64")
        self.assertEqual(original.grad.tolist(), [2, 4, 6])
        self.assertIs(target.to("cuda"), target)

    def test_math_domain_and_internal_index_errors(self):
        for dtype in ("float32", "float64"):
            with self.assertRaisesRegex(ValueError, "大于零"):
                Tensor([1, 0], device="cuda", dtype=dtype).log()
            with self.assertRaises(ZeroDivisionError):
                Tensor([1, 2], device="cuda", dtype=dtype) / [1, 0]
        indices = cuda.storage([-1, 1], dtype="int64")
        with self.assertRaisesRegex(ValueError, "int64"):
            cuda.add(indices, indices)

    def test_two_steps_sgd_adamw_without_gpu_readback_or_cpu_math(self):
        for optimizer_type, options in ((nn.SGD, {"momentum": .9}), (nn.AdamW, {})):
            runs = []
            for device in ("cpu", "cuda"):
                model = nn.Model([nn.Linear(2, 4, seed=7), nn.ReLU(), nn.Linear(4, 1, seed=8)]).to(device, dtype="float32")
                optimizer = optimizer_type(model, learning_rate=.03, **options)
                history = [nn.MSE()(model([[1, 2], [2, 1]]), [[3], [3]]).item()]
                launches = cuda_runtime.get_runtime().launches
                for _ in range(2):
                    if device == "cuda":
                        with patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("训练不得读取显存列表")), patch.object(cpu, "matmul", side_effect=AssertionError("禁止 CPU 回退")):
                            optimizer.zero_grad()
                            loss = nn.MSE()(model([[1, 2], [2, 1]]), [[3], [3]])
                            loss.backward()
                            optimizer.step()
                    else:
                        optimizer.zero_grad()
                        nn.MSE()(model([[1, 2], [2, 1]]), [[3], [3]]).backward()
                        optimizer.step()
                    history.append(nn.MSE()(model([[1, 2], [2, 1]]), [[3], [3]]).item())
                if device == "cuda": self.assertGreater(cuda_runtime.get_runtime().launches, launches)
                self.assertLess(history[-1], history[0])
                self.assertTrue(all(p.device == device and p.grad.device == device for p in model.parameters()))
                for name in optimizer._buffers:
                    self.assertTrue(all(value.device == device for value in getattr(optimizer, name).values()))
                runs.append((history, [p.storage.tolist() for p in model.parameters()]))
            self.assertClose(runs[0][0], runs[1][0])
            for a, b in zip(runs[0][1], runs[1][1]): self.assertClose(a, b)

    def test_existing_optimizer_and_save_load_resume(self):
        with TemporaryDirectory() as folder:
            model = nn.Model([nn.Linear(2, 1, seed=4)])
            optimizer = nn.AdamW(model)
            nn.MSE()(model([[1, 2]]), [[0]]).backward()
            optimizer.step()
            identity = model.parameters()[0]
            model.to("cuda", dtype="float32")
            self.assertIs(model.parameters()[0], identity)
            optimizer.zero_grad()
            nn.MSE()(model([[1, 2]]), [[0]]).backward()
            optimizer.step()
            self.assertEqual(optimizer.averages[identity].device, "cuda")
            model.save(Path(folder)/"model.json")
            optimizer.save(Path(folder)/"optimizer.json")
            restored = nn.Model().load(Path(folder)/"model.json").to("cuda")
            resumed = nn.AdamW(restored).load(Path(folder)/"optimizer.json")
            for target, opt in ((model, optimizer), (restored, resumed)):
                opt.zero_grad()
                nn.MSE()(target([[1, 2]]), [[0]]).backward()
                opt.step()
            for a, b in zip(model.parameters(), restored.parameters()): self.assertClose(a.storage.tolist(), b.storage.tolist())
