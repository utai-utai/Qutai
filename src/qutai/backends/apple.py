"""Apple Silicon GPU：标准库 ctypes 调用自有 Metal 桥接，首次使用时才编译加载。"""

import ctypes as ct
from builtins import sum as scalar_sum
import platform
import subprocess
import sys
from array import array
from collections.abc import Iterable
from functools import lru_cache
from hashlib import sha256
from math import hypot, prod, sqrt
from pathlib import Path
from tempfile import gettempdir

from ..core.shape import broadcast_shape, normalize_axis, reduced_shape, strides, transpose_axes
from .storage import Storage


def _check(success: int, api) -> None:
    if not success: raise RuntimeError("Apple GPU：" + api.q_error().decode())


@lru_cache(maxsize=1)
def _runtime():
    """导入模块不探测设备、不加载系统 GPU 库，也不要求编译工具。"""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("apple 后端需要 Apple Silicon Mac")
    sources = Path(__file__).parent / "apple"
    bridge = sources / "apple.mm"
    shaders = "\n".join(path.read_text() for path in sorted(sources.glob("apple*.metal")))
    fingerprint = sha256(bridge.read_bytes() + shaders.encode()).hexdigest()[:16]
    library = Path(gettempdir()) / "qutai-apple" / fingerprint / "apple.dylib"
    if not library.exists():
        library.parent.mkdir(parents=True, exist_ok=True)
        command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                   "-mmacosx-version-min=13.0", "-framework", "Foundation", "-framework", "Metal", "-framework", "MetalPerformanceShaders",
                   str(bridge), "-o", str(library)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode: raise RuntimeError("Apple 桥接编译失败，请检查 Xcode Command Line Tools：\n" + result.stderr)
    api = ct.CDLL(str(library))
    signatures = {
        "q_error": (ct.c_char_p, []), "q_device_name": (ct.c_char_p, []), "q_dispatches": (ct.c_uint64, []),
        "q_init": (ct.c_int, [ct.c_char_p]), "q_alloc": (ct.c_void_p, [ct.c_uint64]),
        "q_matmul": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32]),
        "q_free": (None, [ct.c_void_p]),
        "q_write": (None, [ct.c_void_p, ct.c_void_p, ct.c_uint64]),
        "q_read": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_uint64]),
        "q_read_range": (ct.c_int, [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_uint64]),
        "q_write_range": (ct.c_int, [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_uint64]),
        "q_copy": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_uint64]), "q_sync": (ct.c_int, []),
        "q_run": (ct.c_int, [ct.c_char_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32), ct.c_uint32,
                             ct.POINTER(ct.c_float), ct.c_uint32, ct.c_uint64]),
    }
    for name, (result_type, arguments) in signatures.items():
        function = getattr(api, name)
        function.restype, function.argtypes = result_type, arguments
    _check(api.q_init(shaders.encode()), api)
    return api


def is_available() -> bool:
    try:
        _runtime()
        return True
    except (OSError, RuntimeError):
        return False


def device_name() -> str:
    return _runtime().q_device_name().decode()


def synchronize() -> None:
    api = _runtime()
    _check(api.q_sync(), api)


class AppleStorage(Storage):
    """拥有一个共享内存 Metal buffer；数值运算仍由 GPU 执行。"""
    device = "apple"

    def __init__(self, values: Iterable[float] = (), *, dtype: str = "float32", size: int | None = None) -> None:
        self._buffer = None
        if dtype not in ("float32", "complex64", "int64"): raise ValueError("apple 支持 float32、complex64 和内部 int64，请显式转换精度")
        self.dtype = dtype
        if dtype == "complex64": values = (part for value in values for part in (complex(value).real, complex(value).imag))
        data = array("q" if dtype == "int64" else "f", values)
        count = len(data) // (2 if dtype == "complex64" else 1)
        if data and size is not None and count != size: raise ValueError("输入元素数量与存储大小不一致")
        self._size = count if size is None else size
        if not 0 <= self._size < 2 ** 32: raise ValueError("apple 后端当前元素数量需要小于 2**32")
        self._api = _runtime()
        self._buffer = self._api.q_alloc(self.nbytes)
        if not self._buffer: raise RuntimeError("Apple GPU：" + self._api.q_error().decode())
        if data:
            self._api.q_write(self._buffer, data.buffer_info()[0], self.nbytes)

    def __del__(self):
        if self._buffer: self._api.q_free(self._buffer)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size * (4 if self.dtype == "float32" else 8)

    def copy(self) -> AppleStorage:
        result = AppleStorage(size=self.size, dtype=self.dtype)
        _check(self._api.q_copy(result._buffer, self._buffer, self.nbytes), self._api)
        return result

    def tolist(self) -> list[float]:
        count = self.size * (2 if self.dtype == "complex64" else 1)
        result = ((ct.c_int64 if self.dtype == "int64" else ct.c_float) * count)()
        _check(self._api.q_read(self._buffer, result, self.nbytes), self._api)
        return [complex(result[i], result[i + 1]) for i in range(0, count, 2)] if self.dtype == "complex64" else list(result)

    def read_bytes(self, offset: int, size: int) -> bytes:
        """显式分块导出，用于二进制 checkpoint。"""
        if not 0 <= offset <= offset + size <= self.nbytes: raise ValueError("读取范围超出存储")
        result = (ct.c_ubyte * size)()
        _check(self._api.q_read_range(self._buffer, offset, result, size), self._api)
        return bytes(result)

    def write_bytes(self, offset: int, content) -> None:
        """显式分块载入，保留原存储对象。"""
        data = bytes(content)
        if not 0 <= offset <= offset + len(data) <= self.nbytes: raise ValueError("写入范围超出存储")
        _check(self._api.q_write_range(self._buffer, offset, data, len(data)), self._api)


def storage(values: Iterable[float] = (), *, dtype: str = "float32") -> AppleStorage:
    return AppleStorage(values, dtype=dtype)


def from_storage(values: Storage, *, dtype: str) -> AppleStorage:
    if dtype not in ("float32", "complex64", "int64"): raise ValueError("apple 支持 float32、complex64 和内部 int64，不支持：" + dtype)
    return values.copy() if isinstance(values, AppleStorage) and values.dtype == dtype else storage(values.tolist(), dtype=dtype)


def copy_into(target: AppleStorage, source: AppleStorage) -> None:
    if (target.device, target.dtype, target.size) != (source.device, source.dtype, source.size):
        raise ValueError("复制存储的设备、精度和元素数量必须相同")
    _check(target._api.q_copy(target._buffer, source._buffer, target.nbytes), target._api)


def _run(name: str, size: int, a: AppleStorage | None = None, b: AppleStorage | None = None,
         c: AppleStorage | None = None, *, meta: tuple = (), scalars: tuple = (), dtype: str | None = None, out: AppleStorage | None = None) -> AppleStorage:
    dtype = dtype or (a.dtype if a is not None else "float32")
    if name == "pointwise" and dtype == "complex64": name = "complex_pointwise"
    result = AppleStorage(size=size, dtype=dtype) if out is None else out
    buffers = (ct.c_void_p * 4)(*(value._buffer if value is not None else None for value in (a, b, c, result)))
    integers = (ct.c_uint32 * (len(meta) + 1))(size, *meta)
    floats = (ct.c_float * max(len(scalars), 1))(*scalars)
    api = result._api
    _check(api.q_run(name.encode(), buffers, integers, len(integers), floats, len(floats), size), api)
    return result


def full(size: int, value: float = 0.0, *, dtype: str = "float32") -> AppleStorage:
    if dtype == "int64": return _run("fill_index", size, meta=(int(value) & 0xffffffff, (int(value) >> 32) & 0xffffffff), dtype=dtype)
    parts = (complex(value).real, complex(value).imag) if dtype == "complex64" else (value,)
    return _run("pointwise", size, meta=(9,), scalars=parts, dtype=dtype)


def add(left: AppleStorage, right: AppleStorage) -> AppleStorage:
    return _run("pointwise", left.size, left, right, meta=(0,))


def subtract(left: AppleStorage, right: AppleStorage) -> AppleStorage:
    return _run("pointwise", left.size, left, right, meta=(1,))


def multiply(left: AppleStorage, right: AppleStorage) -> AppleStorage:
    return _run("pointwise", left.size, left, right, meta=(2,))


def divide(left: AppleStorage, right: AppleStorage) -> AppleStorage:
    return _run("pointwise", left.size, left, right, meta=(3,))


def scale(values: AppleStorage, factor: float) -> AppleStorage:
    return _run("pointwise", values.size, values, meta=(4,), scalars=(factor,))


def negate(values: AppleStorage) -> AppleStorage:
    return _run("pointwise", values.size, values, meta=(5,))


def square(values: AppleStorage) -> AppleStorage:
    return _run("pointwise", values.size, values, meta=(6,))


def relu(values: AppleStorage) -> AppleStorage:
    return _run("pointwise", values.size, values, meta=(7,))


def relu_backward(gradient: AppleStorage, values: AppleStorage) -> AppleStorage:
    return _run("pointwise", gradient.size, gradient, values, meta=(8,))


def broadcast(values: AppleStorage, shape: tuple, target: tuple) -> AppleStorage:
    padded = (1,) * (len(target) - len(shape)) + shape
    steps = tuple(0 if size == 1 else step for size, step in zip(padded, strides(padded), strict=True))
    return _run("reindex", prod(target), values, meta=(len(target), *target, *steps))


def transpose(values: AppleStorage, shape: tuple, axes: tuple | None = None) -> AppleStorage:
    axes = transpose_axes(len(shape), axes)
    target = tuple(shape[axis] for axis in axes)
    source_steps = strides(shape)
    return _run("reindex", values.size, values, meta=(len(shape), *target, *(source_steps[axis] for axis in axes)))


def matmul(left: AppleStorage, right: AppleStorage, left_shape: tuple, right_shape: tuple) -> AppleStorage:
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    left_shape = (1,) * (len(batches) + 2 - len(left_shape)) + left_shape
    right_shape = (1,) * (len(batches) + 2 - len(right_shape)) + right_shape
    rows, inner, columns = left_shape[-2], left_shape[-1], right_shape[-1]
    left_steps = tuple(0 if size == 1 else step for size, step in zip(left_shape[:-2], strides(left_shape)[:-2], strict=True))
    right_steps = tuple(0 if size == 1 else step for size, step in zip(right_shape[:-2], strides(right_shape)[:-2], strict=True))
    if not batches and rows and inner and columns and rows * inner * columns >= 4096:
        result = AppleStorage(size=rows * columns)
        _check(result._api.q_matmul(left._buffer, right._buffer, result._buffer, rows, inner, columns), result._api)
        return result
    return _run("matrix_multiply", prod(batches) * rows * columns, left, right,
                meta=(rows, columns, inner, len(batches), *batches, *left_steps, *right_steps))


def sum(values: AppleStorage, shape: tuple | None = None, axis: int | None = None) -> AppleStorage:
    if axis is None: return _run("reduce_sum", 1, values, meta=(values.size, 1))
    axis = normalize_axis(axis, len(shape))
    return _run("reduce_sum", prod(reduced_shape(shape, axis)), values, meta=(shape[axis], prod(shape[axis + 1:])))


def mean(values: AppleStorage, shape: tuple | None = None, axis: int | None = None) -> AppleStorage:
    count = values.size if axis is None else shape[axis]
    return scale(sum(values, shape, axis), 1 / count)


def unbroadcast(values: AppleStorage, shape: tuple, target: tuple) -> AppleStorage:
    padded = (1,) * (len(shape) - len(target)) + target
    for axis, (size, original) in enumerate(zip(shape, padded, strict=True)):
        if original == 1 and size != 1:
            values = sum(values, shape, axis)
            shape = shape[:axis] + (1,) + shape[axis + 1:]
    return values


def sgd_step(values: AppleStorage, gradient: AppleStorage, velocity: AppleStorage | None, *, learning_rate: float, momentum: float, weight_decay: float) -> tuple:
    direction = add(gradient, scale(values, weight_decay)) if weight_decay else gradient
    updated = None
    if momentum:
        updated = add(scale(velocity, momentum), direction) if velocity is not None else direction.copy()
        direction = updated
    return subtract(values, scale(direction, learning_rate)), updated


def adamw_step(values: AppleStorage, gradient: AppleStorage, average: AppleStorage, square_average: AppleStorage, *, step: int, learning_rate: float, betas: tuple, eps: float, weight_decay: float) -> tuple:
    beta1, beta2 = betas
    average = _run("pointwise", values.size, average, gradient, meta=(10,), scalars=(beta1,))
    square_average = _run("pointwise", values.size, square_average, square(gradient), meta=(10,), scalars=(beta2,))
    result = _run("adamw_update", values.size, values, average, square_average,
                  scalars=(learning_rate, 1 - beta1 ** step, 1 - beta2 ** step, eps, 1 - learning_rate * weight_decay))
    return result, average, square_average



def exp(values: AppleStorage) -> AppleStorage:
    return _run('math_unary', values.size, values, meta=(0,))


def log(values: AppleStorage) -> AppleStorage:
    return _run('math_unary', values.size, values, meta=(1,))


def gelu(values: AppleStorage) -> AppleStorage:
    return _run('math_unary', values.size, values, meta=(2,))


def gelu_backward(gradient: AppleStorage, values: AppleStorage) -> AppleStorage:
    return _run('math_unary', values.size, values, gradient, meta=(3,))


def tanh(values): return _run('math_unary', values.size, values, meta=(5,))
def atan(values): return _run('math_unary', values.size, values, meta=(6,))
def tanh_backward(gradient, values): return _run('math_unary', values.size, values, gradient, meta=(7,))
def atan_backward(gradient, values): return _run('math_unary', values.size, values, gradient, meta=(8,))


def gather(values: AppleStorage, positions) -> AppleStorage:
    indices = positions if isinstance(positions, AppleStorage) else storage(positions, dtype='int64')
    return _run('gather_float' if values.dtype == 'float32' else 'gather_wide', indices.size, values, indices)


def scatter(values: AppleStorage, positions, size: int) -> AppleStorage:
    indices = positions if isinstance(positions, AppleStorage) else storage(positions, dtype='int64')
    return _run('scatter_add', values.size, values, indices, out=full(size))


def concatenate(values: tuple, shapes: tuple, axis: int = 0) -> AppleStorage:
    axis = normalize_axis(axis, len(shapes[0]))
    chunks = [prod(shape[axis:]) for shape in shapes]
    width = scalar_sum(chunks)
    result = AppleStorage(size=scalar_sum(v.size for v in values), dtype=values[0].dtype)
    offset = 0
    for value, chunk in zip(values, chunks, strict=True):
        _run('concat_float' if value.dtype == 'float32' else 'concat_wide', value.size, value,
             meta=(chunk, width, offset), out=result)
        offset += chunk
    return result


def softmax(values: AppleStorage, shape: tuple, axis: int) -> AppleStorage:
    return _run('row_softmax', values.size, values, meta=(shape[axis], prod(shape[axis+1:])))


def argmax(values: AppleStorage, shape: tuple, axis: int | None = None) -> list[int]:
    count, inner = (values.size, 1) if axis is None else (shape[axis], prod(shape[axis+1:]))
    return _run('row_argmax', values.size//count, values, meta=(count, inner), dtype='int64').tolist()


def cross_entropy(values: AppleStorage, shape: tuple, labels: tuple) -> AppleStorage:
    rows = _run('cross_entropy_rows', shape[0], values, storage(labels, dtype='int64'), meta=(shape[1],))
    return mean(rows)


def cross_entropy_backward(gradient: AppleStorage, scores: AppleStorage, shape: tuple, labels: tuple) -> AppleStorage:
    return _run('cross_entropy_grad', scores.size, softmax(scores, shape, 1), storage(labels, dtype='int64'), gradient, meta=(shape[1], shape[0]))


def global_norm(values: tuple) -> float:
    return hypot(*(_run('vector_norm', 1, value, meta=(value.size,)).tolist()[0] for value in values))


def layer_norm(values: AppleStorage, weight: AppleStorage, bias: AppleStorage, width: int, eps: float) -> tuple:
    shape = (values.size//width, width)
    averages = broadcast(mean(values, shape, 1), (shape[0], 1), shape)
    centered = subtract(values, averages)
    inverse = _run('math_unary', shape[0], add(mean(square(centered), shape, 1), full(shape[0], eps)), meta=(4,))
    normalized = multiply(centered, broadcast(inverse, (shape[0], 1), shape))
    return add(multiply(normalized, broadcast(weight, (width,), shape)), broadcast(bias, (width,), shape)), normalized, inverse


def layer_norm_backward(gradient: AppleStorage, normalized: AppleStorage, inverse: AppleStorage, weight: AppleStorage) -> tuple:
    shape = (gradient.size//weight.size, weight.size)
    weighted = multiply(gradient, broadcast(weight, (weight.size,), shape))
    averages = broadcast(mean(weighted, shape, 1), (shape[0], 1), shape)
    projected = broadcast(mean(multiply(weighted, normalized), shape, 1), (shape[0], 1), shape)
    incoming = multiply(subtract(subtract(weighted, averages), multiply(normalized, projected)), broadcast(inverse, (shape[0], 1), shape))
    return incoming, sum(multiply(gradient, normalized), shape, 0), sum(gradient, shape, 0)


def _channel_sum(values: AppleStorage, shape: tuple) -> AppleStorage:
    return sum(sum(values, shape, 0), (shape[1], prod(shape[2:])), 1)


def channel_stats(values: AppleStorage, shape: tuple) -> tuple:
    count = shape[0]*prod(shape[2:])
    average = scale(_channel_sum(values, shape), 1/count)
    centered = subtract(values, broadcast(average, (1,shape[1],1,1), shape))
    return average, scale(_channel_sum(square(centered), shape), 1/count)


def batch_norm2d(values: AppleStorage, weight: AppleStorage, bias: AppleStorage, shape: tuple, averages: AppleStorage, variances: AppleStorage, eps: float) -> tuple:
    channel_shape = (1,shape[1],1,1)
    inverse = _run('math_unary', shape[1], add(variances, full(shape[1], eps)), meta=(4,))
    normalized = multiply(subtract(values, broadcast(averages, channel_shape, shape)), broadcast(inverse, channel_shape, shape))
    result = add(multiply(normalized, broadcast(weight, channel_shape, shape)), broadcast(bias, channel_shape, shape))
    return result, normalized, inverse


def batch_norm2d_backward(gradient: AppleStorage, normalized: AppleStorage, inverse: AppleStorage, weight: AppleStorage, shape: tuple, train: bool) -> tuple:
    weight_gradient = _channel_sum(multiply(gradient, normalized), shape)
    bias_gradient = _channel_sum(gradient, shape)
    channel_shape = (1,shape[1],1,1)
    if train:
        correction = add(broadcast(bias_gradient, channel_shape, shape), multiply(normalized, broadcast(weight_gradient, channel_shape, shape)))
        gradient = subtract(gradient, scale(correction, 1/(shape[0]*prod(shape[2:]))))
    incoming = multiply(gradient, broadcast(multiply(weight, inverse), channel_shape, shape))
    return incoming, weight_gradient, bias_gradient


def _convolution_meta(shape: tuple, weight_shape: tuple, stride: tuple, padding: tuple) -> tuple:
    height = (shape[2]+2*padding[0]-weight_shape[2])//stride[0]+1
    width = (shape[3]+2*padding[1]-weight_shape[3])//stride[1]+1
    return (*shape, weight_shape[0], *weight_shape[2:], *stride, *padding, height, width)


def conv2d(values: AppleStorage, weight: AppleStorage, bias: AppleStorage | None, shape: tuple, weight_shape: tuple, stride: tuple, padding: tuple) -> AppleStorage:
    meta = _convolution_meta(shape, weight_shape, stride, padding)
    return _run('convolution', shape[0]*weight_shape[0]*meta[-2]*meta[-1], values, weight, bias, meta=(*meta, int(bias is not None)))


def conv2d_backward(gradient: AppleStorage, values: AppleStorage, weight: AppleStorage, shape: tuple, weight_shape: tuple, stride: tuple, padding: tuple, *, input_grad: bool = True) -> tuple:
    meta = _convolution_meta(shape, weight_shape, stride, padding)
    incoming = _run('convolution_input_grad', values.size, gradient, weight, meta=meta) if input_grad else full(values.size)
    weights = _run('convolution_weight_grad', weight.size, gradient, values, meta=meta)
    biases = _channel_sum(gradient, (shape[0],weight_shape[0],meta[-2],meta[-1]))
    return incoming, weights, biases


def max_pool2d(values: AppleStorage, shape: tuple, kernel_size: tuple, stride: tuple, padding: tuple) -> tuple:
    meta = _convolution_meta(shape, (shape[1],1,*kernel_size), stride, padding)
    indices = _run('pool_indices', shape[0]*shape[1]*meta[-2]*meta[-1], values, meta=meta, dtype='int64')
    return gather(values, indices), indices


def max_pool2d_backward(gradient: AppleStorage, indices: AppleStorage, size: int) -> AppleStorage:
    return scatter(gradient, indices, size)


def basis_state(qubits: int, *, dtype: str = 'complex64') -> AppleStorage:
    if dtype != 'complex64': raise ValueError('apple 的量子态需要 complex64')
    return _run('zero_state', 2**qubits, dtype=dtype)


def apply_gate(amplitudes: AppleStorage, qubits: int, wires: tuple, matrix: tuple) -> AppleStorage:
    coefficients = storage((value for row in matrix for value in row), dtype='complex64')
    return _run('quantum_gate', amplitudes.size, amplitudes, coefficients, meta=(len(wires), *(qubits-1-wire for wire in wires)))


def probabilities(amplitudes: AppleStorage, qubits: int, wires: tuple) -> AppleStorage:
    shifts = tuple(qubits-1-wire for wire in wires)
    other = tuple(shift for shift in range(qubits) if shift not in shifts)
    return _run('quantum_probabilities', 2**len(wires), amplitudes, meta=(qubits,len(wires),*shifts,*other), dtype='float32')


def collapse(amplitudes: AppleStorage, qubits: int, wires: tuple, outcome: int, probability: float) -> AppleStorage:
    return _run('quantum_collapse', amplitudes.size, amplitudes,
                meta=(qubits,len(wires),*(qubits-1-wire for wire in wires),outcome), scalars=(1/sqrt(probability),))


def inner(left: AppleStorage, right: AppleStorage) -> complex:
    if left.dtype == 'complex64': return _run('inner_complex', 1, left, right, meta=(left.size,)).tolist()[0]
    return complex(_run('inner_real', 1, left, right, meta=(left.size,), scalars=(1,)).tolist()[0])


def real_inner(left: AppleStorage, right: AppleStorage, scale: float = 1.0) -> AppleStorage:
    return _run('inner_complex_real' if left.dtype == 'complex64' else 'inner_real', 1, left, right,
                meta=(left.size,), scalars=(scale,), dtype='float32')


def weighted_sum(values: tuple, weights: AppleStorage) -> AppleStorage:
    result = full(values[0].size, dtype=values[0].dtype)
    for index, value in enumerate(values):
        result = add(result, _run('weighted_complex' if value.dtype == 'complex64' else 'weighted_real', value.size, value, weights, meta=(index,)))
    return result


def probabilities_backward(gradient: AppleStorage, amplitudes: AppleStorage, qubits: int, wires: tuple) -> AppleStorage:
    return _run('quantum_probability_grad', amplitudes.size, amplitudes, gradient,
                meta=(qubits,len(wires),*(qubits-1-wire for wire in wires)))



def add_into(target: AppleStorage, source: AppleStorage) -> None:
    copy_into(target, add(target, source))


def scale_into(target: AppleStorage, factor: float) -> None:
    copy_into(target, scale(target, factor))


def apply_rotation(amplitudes: AppleStorage, qubits: int, wire: int, name: str, angle: AppleStorage, *, adjoint: bool = False, derivative: bool = False) -> AppleStorage:
    """RX/RY/RZ 直接读取设备角度，前向与局部导数均留在 GPU。"""
    return _run("quantum_rotation", amplitudes.size, amplitudes, angle,
                meta=(qubits - 1 - wire, ("RX", "RY", "RZ").index(name), adjoint, derivative))
