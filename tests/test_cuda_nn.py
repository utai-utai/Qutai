from contextlib import contextmanager, ExitStack
from math import prod
from random import Random
import unittest
from unittest.mock import patch

from qutai import Tensor, is_available, nn
from qutai.backends import cpu, cuda
from qutai.backends.storage import Storage


@contextmanager
def gpu_only():
    """核心计算禁止读回显存，也禁止调用 CPU 参考内核。"""
    with ExitStack() as stack:
        stack.enter_context(patch.object(cuda.CUDAStorage, "tolist", side_effect=AssertionError("核心计算不能读取显存列表")))
        for name in ("matmul", "conv2d", "conv2d_backward", "max_pool2d", "max_pool2d_backward",
                     "softmax", "layer_norm", "layer_norm_backward", "channel_stats", "batch_norm2d", "batch_norm2d_backward"):
            stack.enter_context(patch.object(cpu, name, side_effect=AssertionError("CUDA 不能回退 CPU 内核")))
        yield


@unittest.skipUnless(is_available("cuda"), "需要实际 CUDA 设备与 NVRTC")
class CUDANNTests(unittest.TestCase):
    def assertStorageClose(self, actual, expected, dtype):
        if isinstance(expected, tuple):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True): self.assertStorageClose(a, b, dtype)
            return
        self.assertEqual((actual.device, actual.dtype, actual.size), ("cuda", expected.dtype, expected.size))
        tolerance = 4e-5 if dtype == "float32" else 2e-10
        for a, b in zip(actual.tolist(), expected.tolist(), strict=True):
            self.assertLessEqual(abs(a-b), tolerance * (1 + abs(b)))

    def compareKernel(self, name, *args, dtype, **kwargs):
        expected = getattr(cpu, name)(*args, **kwargs)
        inputs = [cuda.from_storage(x, dtype=x.dtype) if isinstance(x, Storage) else x for x in args]
        before = cuda.get_runtime().launches
        with gpu_only(): actual = getattr(cuda, name)(*inputs, **kwargs)
        self.assertGreater(cuda.get_runtime().launches, before)
        self.assertStorageClose(actual, expected, dtype)

    def test_convolution_padding_stride_and_all_local_gradients(self):
        rng = Random(14)
        for dtype in ("float32", "float64"):
            shape, weight_shape = (2, 2, 5, 6), (3, 2, 3, 2)
            values = cpu.storage([rng.uniform(-2, 2) for _ in range(prod(shape))], dtype=dtype)
            weight = cpu.storage([rng.uniform(-2, 2) for _ in range(prod(weight_shape))], dtype=dtype)
            bias = cpu.storage([0.2, -0.1, 0.4], dtype=dtype)
            for stride, padding in (((1, 1), (0, 0)), ((2, 1), (1, 1)), ((2, 3), (1, 0))):
                with self.subTest(dtype=dtype, stride=stride, padding=padding):
                    for b in (bias, None):
                        self.compareKernel("conv2d", values, weight, b, shape, weight_shape, stride, padding, dtype=dtype)
                    rows = (shape[2] + 2*padding[0] - weight_shape[2]) // stride[0] + 1
                    columns = (shape[3] + 2*padding[1] - weight_shape[3]) // stride[1] + 1
                    gradient = cpu.storage([rng.uniform(-1, 1) for _ in range(shape[0]*weight_shape[0]*rows*columns)], dtype=dtype)
                    for input_grad in (True, False):
                        self.compareKernel("conv2d_backward", gradient, values, weight, shape, weight_shape, stride, padding,
                                           input_grad=input_grad, dtype=dtype)

    def test_pool_ties_padding_overlapping_and_index_precision(self):
        for dtype in ("float32", "float64"):
            values = cpu.storage([1, 1, 0, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3] * 4, dtype=dtype)
            shape = (2, 2, 4, 4)
            for kernel, stride, padding in (((2, 2), (1, 1), (0, 0)), ((3, 2), (1, 1), (1, 1))):
                with self.subTest(dtype=dtype, kernel=kernel):
                    self.compareKernel("max_pool2d", values, shape, kernel, stride, padding, dtype=dtype)
                    pooled, indices = cpu.max_pool2d(values, shape, kernel, stride, padding)
                    gradient = cpu.storage([0.25 - i*0.03125 for i in range(pooled.size)], dtype=dtype)
                    self.compareKernel("max_pool2d_backward", gradient, indices, values.size, dtype=dtype)

    def test_softmax_every_axis_and_autograd(self):
        rng = Random(42)
        data, incoming = [rng.uniform(-8, 8) for _ in range(24)], [rng.uniform(-2, 2) for _ in range(24)]
        for dtype in ("float32", "float64"):
            for axis in (0, 1, 2, -1):
                with self.subTest(dtype=dtype, axis=axis):
                    cpu_x = Tensor(data, shape=(2, 3, 4), dtype=dtype, requires_grad=True)
                    reference = nn.Softmax(axis)(cpu_x)
                    (reference * Tensor(incoming, shape=cpu_x.shape, dtype=dtype)).sum().backward()
                    gpu_x = Tensor(data, shape=cpu_x.shape, dtype=dtype, device="cuda", requires_grad=True)
                    factors = Tensor(incoming, shape=cpu_x.shape, dtype=dtype, device="cuda")
                    with gpu_only():
                        actual = nn.Softmax(axis)(gpu_x)
                        (actual * factors).sum().backward()
                    self.assertStorageClose(actual.storage, reference.storage, dtype)
                    self.assertStorageClose(gpu_x.grad.storage, cpu_x.grad.storage, dtype)

    def test_normalization_statistics_forward_and_backward(self):
        rng = Random(4)
        for dtype in ("float32", "float64"):
            shape = (2, 3, 4, 5)
            values = cpu.storage([rng.uniform(-2, 2) for _ in range(prod(shape))], dtype=dtype)
            gradient = cpu.storage([rng.uniform(-1, 1) for _ in range(values.size)], dtype=dtype)
            weight, bias = cpu.storage([0.4, -0.3, 0.9], dtype=dtype), cpu.storage([0.1, -0.4, 0.7], dtype=dtype)
            self.compareKernel("channel_stats", values, shape, dtype=dtype)
            averages, variances = cpu.channel_stats(values, shape)
            self.compareKernel("batch_norm2d", values, weight, bias, shape, averages, variances, 1e-5, dtype=dtype)
            _, normalized, inverse = cpu.batch_norm2d(values, weight, bias, shape, averages, variances, 1e-5)
            for train in (True, False):
                self.compareKernel("batch_norm2d_backward", gradient, normalized, inverse, weight, shape, train, dtype=dtype)
            self.compareKernel("batch_norm2d", cpu.storage([], dtype=dtype), weight, bias, (0, 3, 4, 5), averages, variances, 1e-5, dtype=dtype)
            weight, bias = cpu.storage([0.1, 0.9, -0.2, 0.3, 0.6], dtype=dtype), cpu.full(5, dtype=dtype)
            self.compareKernel("layer_norm", values, weight, bias, 5, 1e-5, dtype=dtype)
            _, normalized, inverse = cpu.layer_norm(values, weight, bias, 5, 1e-5)
            self.compareKernel("layer_norm_backward", gradient, normalized, inverse, weight, dtype=dtype)

    def test_cnn_batchnorm_pool_layernorm_two_adamw_updates(self):
        rng = Random(8)
        data = [rng.uniform(-2, 2) for _ in range(2*1*6*7)]
        targets = [[0.3, -0.2, 0.7], [-0.4, 0.5, 0.1]]
        # BatchNorm 对常数平移不敏感；不用冗余卷积偏置，避免 Adam 放大理论零梯度的舍入噪声。
        for dtype in ("float32", "float64"):
            models = []
            for device in ("cpu", "cuda"):
                model = nn.Model([nn.Conv2d(1, 2, 2, bias=False, seed=2), nn.BatchNorm2d(2), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Flatten(), nn.LayerNorm(12, eps=0.01), nn.Linear(12, 3, seed=3)]).to(device, dtype=dtype)
                models.append((model, nn.AdamW(model, learning_rate=0.002)))
            for _ in range(2):
                expected_model, expected_opt = models[0]
                expected_x = Tensor(data, shape=(2, 1, 6, 7), dtype=dtype, requires_grad=True)
                expected_opt.zero_grad()
                expected_loss = nn.MSE()(expected_model(expected_x), targets)
                expected_loss.backward()
                model, optimizer = models[1]
                x = Tensor(data, shape=expected_x.shape, dtype=dtype, device="cuda", requires_grad=True)
                with gpu_only():
                    optimizer.zero_grad()
                    loss = nn.MSE()(model(x), targets)
                    loss.backward()
                    optimizer.step()
                expected_opt.step()
                self.assertStorageClose(loss.storage, expected_loss.storage, dtype)
                self.assertStorageClose(x.grad.storage, expected_x.grad.storage, dtype)
                for p, q in zip(model.parameters(), expected_model.parameters(), strict=True):
                    self.assertStorageClose(p.storage, q.storage, dtype)
                    self.assertStorageClose(p.grad.storage, q.grad.storage, dtype)
                self.assertStorageClose(model.layers[1].running_mean, expected_model.layers[1].running_mean, dtype)
                self.assertStorageClose(model.layers[1].running_var, expected_model.layers[1].running_var, dtype)


if __name__ == "__main__": unittest.main()
