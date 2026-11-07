"""单流 CUDA 后端；仅在实际使用设备时加载 Driver/NVRTC。"""

from array import array
from collections.abc import Iterable
from ctypes import c_uint64, c_longlong, c_float, c_double
from math import prod
from pathlib import Path
from secrets import randbits
import struct

from .precision import matmul_precision

from ..core.shape import broadcast_shape, normalize_axis, strides, transpose_axes
from .storage import Storage
from .cuda_runtime import get_runtime


_TYPES = {"float32": ("f", 4), "float64": ("d", 8), "complex64": ("f", 8), "complex128": ("d", 16), "int64": ("q", 8), "bfloat16": ("H", 2)}


class CUDAStorage(Storage):
    """拥有独立显存；tolist 是显式同步读取出口。"""
    device = "cuda"

    def __init__(self, size: int, dtype: str = "float32") -> None:
        if dtype not in _TYPES: raise ValueError(f"CUDA 尚不支持精度：{dtype}")
        if size < 0: raise ValueError("存储大小不能为负数")
        self.dtype, self._size = dtype, size
        self.runtime = get_runtime()
        self.ptr = self.runtime.allocate(self.nbytes)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self.size * _TYPES[self.dtype][1]

    def __len__(self) -> int:
        return self.size

    def __del__(self) -> None:
        if getattr(self, "ptr", 0):
            self.runtime.free(self.ptr)
            self.ptr = 0

    def copy(self):
        result = CUDAStorage(self.size, self.dtype)
        self.runtime.copy(result.ptr, self.ptr, self.nbytes)
        return result

    def tolist(self):
        parts = 2 if self.dtype.startswith("complex") else 1
        buffer = array(_TYPES[self.dtype][0], [0]) * (self.size * parts)
        self.runtime.download(self.ptr, buffer)
        if self.dtype == "bfloat16":
            return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in buffer]
        return [complex(buffer[i], buffer[i + 1]) for i in range(0, len(buffer), 2)] if parts == 2 else list(buffer)


def _p(value):
    return c_uint64(value.ptr if value is not None else 0)


def _i(value):
    return c_longlong(value)


def _s(value, dtype):
    return (c_float if dtype in ("float32", "complex64", "bfloat16") else c_double)(value)


def _launch(name, size, *arguments, dtype="float32", source="tensor.cu"):
    if dtype == "bfloat16" and source != "tensor.cu":
        raise ValueError("此算子尚未实现 BF16 存储；请转换为 float32")
    real = "float" if dtype in ("float32", "complex64", "bfloat16") else "double"
    options = (f"-DREAL={real}",) + (("-DBFLOAT16=1",) if dtype == "bfloat16" else ())
    get_runtime().launch(Path(__file__).parent / "kernels" / source, name, size, list(arguments), options=options)


def is_available() -> bool:
    try:
        get_runtime()
        return True
    except (OSError, RuntimeError):
        return False


def synchronize() -> None:
    get_runtime().synchronize()


def storage(values: Iterable = (), *, dtype="float64"):
    if isinstance(values, Storage): return from_storage(values, dtype=dtype)
    if dtype not in _TYPES: raise ValueError(f"CUDA 尚不支持精度：{dtype}")
    parts = 2 if dtype.startswith("complex") else 1
    if parts == 2: values = (p for x in values for p in (complex(x).real, complex(x).imag))
    if dtype == "bfloat16":
        # 主机输入按 round-to-nearest-even 编码；设备间转换由 cast 内核完成。
        bits = (struct.unpack("<I", struct.pack("<f", value))[0] for value in values)
        values = ((value >> 16) | 0x40 if value & 0x7FFFFFFF > 0x7F800000 else (value + 0x7FFF + ((value >> 16) & 1)) >> 16 for value in bits)
    buffer = array(_TYPES[dtype][0], values)
    if len(buffer) == 1 and dtype in ("float32", "float64", "int64"):
        # 标量由同流 fill 排队，避免很小的 HtoD 复制等待此前全部计算。
        return full(1, buffer[0], dtype=dtype)
    result = CUDAStorage(len(buffer) // parts, dtype)
    result.runtime.upload(result.ptr, buffer)
    return result


def full(size, value=0.0, *, dtype="float64"):
    result = CUDAStorage(size, dtype)
    if dtype == "int64":
        _launch("fill_int", size, _p(result), _i(size), _i(int(value)))
    else:
        _launch("fill", size, _p(result), _i(size), _s(complex(value).real, dtype), _s(complex(value).imag, dtype), _i(2 if dtype.startswith("complex") else 1), dtype=dtype)
    return result


def from_storage(values: Storage, *, dtype):
    if values.device != "cuda": return storage(values.tolist(), dtype=dtype)
    if values.dtype == dtype: return values.copy()
    if values.dtype.startswith("complex") and not dtype.startswith("complex"): raise ValueError("复数存储不能隐式丢弃虚部")
    result = CUDAStorage(values.size, dtype)
    source_code = list(_TYPES).index(values.dtype)
    target_code = list(_TYPES).index(dtype)
    _launch("cast", values.size, _p(values), _p(result), _i(values.size), _i(source_code), _i(target_code))
    return result


def copy_into(target, source):
    if (target.device, target.dtype, target.size) != (source.device, source.dtype, source.size): raise ValueError("复制存储的设备、精度与元素数量必须相同")
    target.runtime.copy(target.ptr, source.ptr, target.nbytes)


def add_into(target, source):
    """同形状存储原地累加；每个线程只读写自己的元素。"""
    if (target.device, target.dtype, target.size) != (source.device, source.dtype, source.size):
        raise ValueError("累加存储的设备、精度与元素数量必须相同")
    if target.dtype == "int64": raise ValueError("索引存储不支持梯度累加")
    count = target.size * (2 if target.dtype.startswith("complex") else 1)
    _launch("binary", count, _p(target), _p(source), _p(target), _i(count), _i(0), _p(None), dtype=target.dtype)


def scale_into(values, factor):
    """独立元素的原地缩放，供梯度裁剪复用现有存储。"""
    if values.dtype == "int64": raise ValueError("索引存储不支持数值缩放")
    count = values.size * (2 if values.dtype.startswith("complex") else 1)
    _launch("unary", count, _p(values), _p(None), _p(values), _i(count), _i(2), _s(factor, values.dtype), _p(None), dtype=values.dtype)


def _unary(values, operation, factor=0.0, gradient=None):
    if values.dtype == "int64" or (values.dtype.startswith("complex") and operation not in (0, 2)):
        raise ValueError("此 CUDA 一元运算只支持实数浮点存储")
    result = CUDAStorage(values.size, values.dtype)
    count = values.size * (2 if values.dtype.startswith("complex") else 1)
    status = full(1, dtype="int64") if operation == 8 else None
    _launch("unary", count, _p(values), _p(gradient), _p(result), _i(count), _i(operation), _s(factor, values.dtype), _p(status), dtype=values.dtype)
    if status is not None:
        flag = array("q", [0])
        status.runtime.download(status.ptr, flag)
        if flag[0]: raise ValueError("自然对数的输入必须大于零")
    return result


def _binary(left, right, operation):
    if left.dtype == "int64": raise ValueError("内部 int64 索引存储不支持数学运算")
    if (left.size, left.dtype) != (right.size, right.dtype): raise ValueError("逐元素运算要求相同元素数量和精度")
    result = CUDAStorage(left.size, left.dtype)
    count = left.size * (2 if left.dtype.startswith("complex") else 1)
    if left.dtype.startswith("complex") and operation >= 2: raise ValueError("复数乘除请使用量子内核")
    status = full(1, dtype="int64") if operation == 3 else None
    _launch("binary", count, _p(left), _p(right), _p(result), _i(count), _i(operation), _p(status), dtype=left.dtype)
    if status is not None:
        flag = array("q", [0])
        status.runtime.download(status.ptr, flag)
        if flag[0]: raise ZeroDivisionError("除数不能为零")
    return result


def add(left, right): return _binary(left, right, 0)
def subtract(left, right): return _binary(left, right, 1)
def multiply(left, right): return _binary(left, right, 2)
def divide(left, right): return _binary(left, right, 3)
def negate(values): return _unary(values, 0)
def square(values): return _unary(values, 1)
def scale(values, factor): return _unary(values, 2, factor)
def relu(values): return _unary(values, 3)
def relu_backward(gradient, values): return _unary(values, 4, gradient=gradient)
def gelu(values): return _unary(values, 5)
def gelu_backward(gradient, values): return _unary(values, 6, gradient=gradient)
def exp(values): return _unary(values, 7)
def log(values): return _unary(values, 8)


def gather(values, positions):
    positions = storage(positions, dtype="int64")
    result = CUDAStorage(positions.size, values.dtype)
    _launch("gather", result.size, _p(values), _p(positions), _p(result), _i(result.size), _i(_TYPES[values.dtype][1]))
    return result


def scatter(values, positions, size):
    if values.dtype == "int64": raise ValueError("内部 int64 索引存储不支持梯度累加")
    if values.dtype == "bfloat16":
        return from_storage(scatter(from_storage(values, dtype="float32"), positions, size), dtype="bfloat16")
    positions = storage(positions, dtype="int64")
    result = full(size, dtype=values.dtype)
    parts = 2 if values.dtype.startswith("complex") else 1
    _launch("scatter", values.size * parts, _p(values), _p(positions), _p(result), _i(values.size * parts), _i(parts), dtype=values.dtype)
    return result


def broadcast(values, shape, target):
    if shape == target: return values
    padded = (1,) * (len(target) - len(shape)) + shape
    steps = tuple(0 if length == 1 else step for length, step in zip(padded, strides(padded)))
    meta = storage((*target, *steps), dtype="int64")
    result = CUDAStorage(prod(target), values.dtype)
    _launch("broadcast", result.size, _p(values), _p(result), _p(meta), _i(result.size), _i(len(target)), _i(_TYPES[values.dtype][1]), dtype=values.dtype)
    return result


def transpose(values, shape, axes=None):
    axes = transpose_axes(len(shape), axes)
    target = tuple(shape[a] for a in axes)
    steps = strides(shape)
    meta = storage((*target, *(steps[a] for a in axes)), dtype="int64")
    result = CUDAStorage(values.size, values.dtype)
    _launch("broadcast", result.size, _p(values), _p(result), _p(meta), _i(result.size), _i(len(shape)), _i(_TYPES[values.dtype][1]), dtype=values.dtype)
    return result


def sum(values, shape=None, axis=None):
    if values.dtype not in ("float32", "float64", "bfloat16"): raise ValueError("CUDA sum 目前只支持实数浮点存储")
    if axis is None:
        outer, width, inner = 1, values.size, 1
    else:
        axis = normalize_axis(axis, len(shape))
        outer, width, inner = prod(shape[:axis]), shape[axis], prod(shape[axis + 1:])
    result = CUDAStorage(outer * inner, values.dtype)
    _launch("reduce_sum", result.size, _p(values), _p(result), _i(outer), _i(width), _i(inner), dtype=values.dtype)
    return result


def mean(values, shape=None, axis=None):
    count = values.size if axis is None else shape[normalize_axis(axis, len(shape))]
    if not count: raise ValueError("平均值的元素数量必须大于零")
    return scale(sum(values, shape, axis), 1 / count)


def unbroadcast(values, shape, target):
    current = shape
    for axis, (length, original) in enumerate(zip(shape, (1,) * (len(shape) - len(target)) + target)):
        if original == 1 and length != 1:
            values = sum(values, current, axis)
            current = current[:axis] + (1,) + current[axis + 1:]
    return values


def memory_stats(): return get_runtime().memory_stats()
def reset_peak_memory_stats(): get_runtime().reset_peak_memory_stats()
def empty_cache(): get_runtime().empty_cache()


def normal(size, mean=0.0, std=1.0, *, seed=None, dtype="float32"):
    if dtype not in ("float32", "float64"): raise ValueError("随机初始化只支持实数浮点存储")
    result = CUDAStorage(size, dtype)
    seed = randbits(63) if seed is None else int(seed) & ((1 << 63) - 1)
    _launch("normal_fill", size, _p(result), _i(size), _i(seed), _s(mean,dtype), _s(std,dtype), dtype=dtype, source="gemm.cu")
    return result


def gemm(left, right, rows, inner, cols, *, batch=1, transpose_left=False, transpose_right=False, left_stride=None, right_stride=None, out_dtype=None, out=None, accumulate=False):
    """矩阵乘法；BF16 输入使用 FP32 累加，out 可直接承接或累加结果。"""
    if not isinstance(left, CUDAStorage) or not isinstance(right, CUDAStorage):
        raise ValueError("GEMM 输入必须是 CUDA 存储")
    allowed = ("float32", "bfloat16", "float64")
    if left.dtype not in allowed or right.dtype not in allowed:
        raise ValueError("GEMM 只支持 float32、bfloat16 和 float64")
    double = left.dtype == "float64" or right.dtype == "float64"
    if double and left.dtype != right.dtype:
        raise ValueError("float64 GEMM 的两个输入必须同精度")
    dtype = ("float64" if double else "float32") if out_dtype is None else out_dtype
    if dtype not in (("float64",) if double else ("float32", "bfloat16")):
        raise ValueError("GEMM 输出精度与输入不兼容")
    if any(type(x) is not int or x < 0 for x in (rows, inner, cols, batch)):
        raise ValueError("GEMM 维度必须是非负整数")
    sa = rows * inner if left_stride is None else left_stride
    sb = inner * cols if right_stride is None else right_stride
    if any(type(x) is not int or x < 0 for x in (sa, sb)):
        raise ValueError("GEMM batch stride 必须是非负整数")
    if batch and (left.size < (batch - 1) * sa + rows * inner or right.size < (batch - 1) * sb + inner * cols):
        raise ValueError("GEMM 输入存储不足")
    if accumulate and out is None:
        raise ValueError("GEMM 累加必须指定 out")
    size = batch * rows * cols
    if out is not None:
        if not isinstance(out, CUDAStorage) or out.dtype != dtype or out.size != size:
            raise ValueError("GEMM out 的设备、精度或大小不匹配")
        for value in (left, right):
            if out.nbytes and value.nbytes and out.ptr < value.ptr + value.nbytes and value.ptr < out.ptr + out.nbytes:
                raise ValueError("GEMM out 不能与输入共享存储")
    result = out if out is not None else CUDAStorage(size, dtype)
    if not size:
        return result
    mixed = not double and (left.dtype == "bfloat16" or right.dtype == "bfloat16" or matmul_precision() == "bfloat16")
    runtime = get_runtime()
    if mixed and runtime.compute_capability < (8, 0):
        raise RuntimeError("BF16 Tensor Core 需要 SM80 或更新设备")
    options = ["-DREAL=" + ("double" if double else "float")]
    for macro, value in (("A_BF16", left.dtype), ("B_BF16", right.dtype), ("OUT_BF16", dtype)):
        if value == "bfloat16":
            options.append("-D" + macro)
    tile_m = tile_n = 64 if mixed else 32
    name = "gemm_bf16" if mixed else "gemm_tiled"
    block = (128, 1, 1) if mixed else (16, 16, 1)
    if mixed and inner >= 128:
        if (rows >= 1024 and cols >= 1024) or (rows >= 128 and cols >= 8192):
            name, tile_m, tile_n, block = "gemm_bf16_wide", 128, 128, (256, 1, 1)
        elif 128 <= rows <= 256 and inner >= 8192 and cols >= 1024:
            name, tile_m, tile_n, block = "gemm_bf16_long", 128, 64, (256, 1, 1)
    args = [_p(left), _p(right), _p(result), *(_i(x) for x in (rows, inner, cols, sa, sb, transpose_left, transpose_right, accumulate))]
    runtime.launch(Path(__file__).parent / "kernels/gemm.cu", name, size, args,
                   options=tuple(options), grid=((cols + tile_n - 1) // tile_n, (rows + tile_m - 1) // tile_m, batch),
                   block=block)
    if mixed:
        runtime.tensor_core_launches += 1
    return result


def matmul(left, right, left_shape, right_shape):
    if left.dtype not in ("float32", "float64", "bfloat16"): raise ValueError("CUDA matmul 目前只支持实数浮点存储")
    if len(left_shape) == 1: left_shape = (1, *left_shape)
    if len(right_shape) == 1: right_shape = (*right_shape, 1)
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    a = (1,) * (len(batches) + 2 - len(left_shape)) + left_shape
    b = (1,) * (len(batches) + 2 - len(right_shape)) + right_shape
    count = prod(batches)
    if prod(a[:-2]) in (1, count) and prod(b[:-2]) in (1, count):
        return gemm(left, right, a[-2], a[-1], b[-1], batch=count, left_stride=0 if prod(a[:-2]) == 1 else None, right_stride=0 if prod(b[:-2]) == 1 else None, out_dtype=left.dtype)
    steps_a, steps_b = strides(a), strides(b)
    meta = storage((*batches, *(0 if a[d] == 1 else steps_a[d] for d in range(len(batches))), *(0 if b[d] == 1 else steps_b[d] for d in range(len(batches)))), dtype="int64")
    rows, inner, columns = a[-2], a[-1], b[-1]
    result = CUDAStorage(prod(batches) * rows * columns, left.dtype)
    _launch("matmul", result.size, _p(left), _p(right), _p(result), _p(meta), _i(result.size), _i(rows), _i(inner), _i(columns), _i(len(batches)), dtype=left.dtype)
    return result


def concatenate(values, shapes, axis=0):
    axis = normalize_axis(axis, len(shapes[0]))
    sizes = [prod(shape[axis:]) for shape in shapes]
    outer = prod(shapes[0][:axis])
    total = 0
    for size in sizes: total += size
    result = CUDAStorage(outer * total, values[0].dtype)
    start = 0
    for data, size in zip(values, sizes):
        _launch("concatenate", outer * size, _p(data), _p(result), _i(outer * size), _i(size), _i(total), _i(start), _i(_TYPES[data.dtype][1]))
        start += size
    return result


def argmax(values, shape, axis=None):
    if values.dtype not in ("float32", "float64", "bfloat16"): raise ValueError("CUDA argmax 只支持实数浮点存储")
    if axis is None: outer, width, inner = 1, values.size, 1
    else:
        axis = normalize_axis(axis, len(shape))
        outer, width, inner = prod(shape[:axis]), shape[axis], prod(shape[axis + 1:])
    if not width: raise ValueError("argmax 的计算轴不能为空")
    result = CUDAStorage(outer * inner, "int64")
    _launch("argmax", result.size, _p(values), _p(result), _i(result.size), _i(width), _i(inner), dtype=values.dtype)
    return result.tolist()


def cross_entropy(values, shape, labels):
    rows, columns = shape
    labels = storage(labels, dtype="int64")
    losses = CUDAStorage(rows, values.dtype)
    _launch("cross_entropy", rows, _p(values), _p(labels), _p(losses), _i(rows), _i(columns), dtype=values.dtype)
    return mean(losses)


def cross_entropy_backward(gradient, scores, shape, labels):
    rows, columns = shape
    labels = storage(labels, dtype="int64")
    result = softmax(scores, shape, 1)
    _launch("cross_entropy_backward", result.size, _p(result), _p(gradient), _p(labels), _i(rows), _i(columns), dtype=scores.dtype)
    return result


def global_norm(values):
    if not values: return 0.0
    norms = []
    for data in values:
        if data.dtype not in ("float32", "float64", "bfloat16"): raise ValueError("梯度范数只支持实数浮点存储")
        blocks = min(4096, max(1, (data.size + 1023) // 1024))
        output = CUDAStorage(blocks, "float64")
        _launch("norm_blocks", blocks * 256, _p(data), _p(output), _i(data.size), dtype=data.dtype)
        norms.append(output)
    combined = concatenate(tuple(norms), tuple((value.size,) for value in norms))
    result = CUDAStorage(1, "float64")
    _launch("tensor_norm", 1, _p(combined), _p(result), _i(combined.size), dtype="float64")
    return result.tolist()[0]


def sgd_step(values, gradient, velocity, *, learning_rate, momentum, weight_decay):
    direction = add(gradient, scale(values, weight_decay)) if weight_decay else gradient
    updated = None
    if momentum:
        updated = add(scale(velocity, momentum), direction) if velocity is not None else direction.copy()
        direction = updated
    return subtract(values, scale(direction, learning_rate)), updated


def adamw_step_into(values, gradient, average, square_average, *, step, learning_rate, betas, eps, weight_decay):
    beta1, beta2 = betas
    _launch("adamw", values.size, _p(values), _p(gradient), _p(average), _p(square_average), _p(values), _p(average), _p(square_average), _i(values.size), *(_s(x, values.dtype) for x in (learning_rate, beta1, beta2, 1-beta1**step, 1-beta2**step, eps, weight_decay)), dtype=values.dtype)
    return values, average, square_average


def adamw_step(values, gradient, average, square_average, *, step, learning_rate, betas, eps, weight_decay):
    result, new_average, new_square = (CUDAStorage(values.size, values.dtype) for _ in range(3))
    beta1, beta2 = betas
    _launch("adamw", values.size, _p(values), _p(gradient), _p(average), _p(square_average), _p(result), _p(new_average), _p(new_square), _i(values.size), *(_s(x, values.dtype) for x in (learning_rate, beta1, beta2, 1-beta1**step, 1-beta2**step, eps, weight_decay)), dtype=values.dtype)
    return result, new_average, new_square


# 直接并入 cuda.py；仅调用已有存储和启动工具，无循环导入。

def softmax(values, shape, axis):
    """每组沿指定轴执行稳定 Softmax。"""
    axis %= len(shape)
    width, inner = shape[axis], prod(shape[axis + 1:])
    groups = values.size // width
    result = CUDAStorage(values.size, dtype=values.dtype)
    _launch("softmax_forward", groups, _p(result), _p(values), _i(groups), _i(width), _i(inner), dtype=values.dtype, source="nn.cu")
    return result


def _conv_dimensions(shape, weight_shape, stride, padding):
    N, C, H, W = shape
    O, _, KH, KW = weight_shape
    SH, SW = stride
    PH, PW = padding
    OH, OW = (H + 2 * PH - KH) // SH + 1, (W + 2 * PW - KW) // SW + 1
    return N, C, H, W, O, KH, KW, OH, OW, SH, SW, PH, PW


def conv2d(values, weight, bias, shape, weight_shape, stride, padding):
    """NCHW / OIHW 卷积，直接读取显存。"""
    dims = _conv_dimensions(shape, weight_shape, stride, padding)
    N, C, H, W, O, KH, KW, OH, OW, SH, SW, PH, PW = dims
    result = CUDAStorage(N * O * OH * OW, dtype=values.dtype)
    _launch("conv_forward", result.size, _p(result), _p(values), _p(weight), _p(bias) if bias is not None else _i(0),
            _i(result.size), *(_i(d) for d in dims[1:]), _i(bias is not None), dtype=values.dtype, source="nn.cu")
    return result


def conv2d_backward(gradient, values, weight, shape, weight_shape, stride, padding, *, input_grad=True):
    """分别按输入元素、卷积核元素、通道累加局部梯度。"""
    dims = _conv_dimensions(shape, weight_shape, stride, padding)
    N, C, H, W, O, KH, KW, OH, OW, SH, SW, PH, PW = dims
    dx = CUDAStorage(values.size, dtype=gradient.dtype) if input_grad else full(values.size, dtype=gradient.dtype)
    dw = CUDAStorage(weight.size, dtype=gradient.dtype)
    db = CUDAStorage(O, dtype=gradient.dtype)
    if input_grad:
        _launch("conv_input_backward", dx.size, _p(dx), _p(gradient), _p(weight), _i(dx.size),
                *(_i(d) for d in dims[1:]), dtype=gradient.dtype, source="nn.cu")
    _launch("conv_weight_backward", dw.size, _p(dw), _p(gradient), _p(values), _i(dw.size),
            *(_i(d) for d in dims), dtype=gradient.dtype, source="nn.cu")
    _launch("conv_bias_backward", O, _p(db), _p(gradient), _i(N), _i(O), _i(OH * OW), dtype=gradient.dtype, source="nn.cu")
    return dx, dw, db


def max_pool2d(values, shape, kernel_size, stride, padding):
    """池化保存 int64 显存位置，并列最大值保留第一个。"""
    N, C, H, W = shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding
    OH, OW = (H + 2 * PH - KH) // SH + 1, (W + 2 * PW - KW) // SW + 1
    result = CUDAStorage(N * C * OH * OW, dtype=values.dtype)
    indices = CUDAStorage(result.size, dtype="int64")
    _launch("pool_forward", result.size, _p(result), _p(indices), _p(values), _i(result.size),
            *(_i(d) for d in (H, W, KH, KW, OH, OW, SH, SW, PH, PW)), dtype=values.dtype, source="nn.cu")
    return result, indices


def max_pool2d_backward(gradient, indices, size):
    """重叠窗口的梯度以设备原子加法累加。"""
    result = full(size, dtype=gradient.dtype)
    _launch("pool_backward", gradient.size, _p(result), _p(gradient), _p(indices), _i(gradient.size), dtype=gradient.dtype, source="nn.cu")
    return result


def channel_stats(values, shape):
    """按通道统计批次和空间上的均值与总体方差。"""
    N, C, H, W = shape
    averages, variances = CUDAStorage(C, dtype=values.dtype), CUDAStorage(C, dtype=values.dtype)
    _launch("channel_stats_forward", C, _p(averages), _p(variances), _p(values), _i(N), _i(C), _i(H * W), dtype=values.dtype, source="nn.cu")
    return averages, variances


def batch_norm2d(values, weight, bias, shape, averages, variances, eps):
    N, C, H, W = shape
    result, normalized = CUDAStorage(values.size, dtype=values.dtype), CUDAStorage(values.size, dtype=values.dtype)
    inverse = CUDAStorage(C, dtype=values.dtype)
    _launch("batch_norm_inverse", C, _p(inverse), _p(variances), _i(C), _s(eps, values.dtype), dtype=values.dtype, source="nn.cu")
    _launch("batch_norm_forward", values.size, _p(result), _p(normalized), _p(inverse), _p(values), _p(weight), _p(bias),
            _p(averages), _p(variances), _i(values.size), _i(C), _i(H * W), _s(eps, values.dtype), dtype=values.dtype, source="nn.cu")
    return result, normalized, inverse


def batch_norm2d_backward(gradient, normalized, inverse, weight, shape, train):
    N, C, H, W = shape
    dx = CUDAStorage(gradient.size, dtype=gradient.dtype)
    dw, db = CUDAStorage(C, dtype=gradient.dtype), CUDAStorage(C, dtype=gradient.dtype)
    _launch("norm_parameter_backward", C, _p(dw), _p(db), _p(gradient), _p(normalized), _i(N), _i(C), _i(H * W), dtype=gradient.dtype, source="nn.cu")
    _launch("batch_norm_input_backward", dx.size, _p(dx), _p(gradient), _p(normalized), _p(inverse), _p(weight), _p(dw), _p(db),
            _i(dx.size), _i(C), _i(H * W), _i(N * H * W), _i(train), dtype=gradient.dtype, source="nn.cu")
    return dx, dw, db


def layer_norm(values, weight, bias, width, eps):
    groups = values.size // width
    result, normalized = CUDAStorage(values.size, dtype=values.dtype), CUDAStorage(values.size, dtype=values.dtype)
    inverse = CUDAStorage(groups, dtype=values.dtype)
    _launch("layer_norm_forward", groups, _p(result), _p(normalized), _p(inverse), _p(values), _p(weight), _p(bias),
            _i(groups), _i(width), _s(eps, values.dtype), dtype=values.dtype, source="nn.cu")
    return result, normalized, inverse


def layer_norm_backward(gradient, normalized, inverse, weight):
    width, groups = weight.size, inverse.size
    dx = CUDAStorage(gradient.size, dtype=gradient.dtype)
    dw, db = CUDAStorage(width, dtype=gradient.dtype), CUDAStorage(width, dtype=gradient.dtype)
    _launch("layer_norm_input_backward", groups, _p(dx), _p(gradient), _p(normalized), _p(inverse), _p(weight), _i(groups), _i(width), dtype=gradient.dtype, source="nn.cu")
    _launch("norm_parameter_backward", width, _p(dw), _p(db), _p(gradient), _p(normalized), _i(groups), _i(width), _i(1), dtype=gradient.dtype, source="nn.cu")
    return dx, dw, db


# Merge these functions into backends/cuda.py; all helpers are defined there.

def _quantum_masks(qubits: int, wires: tuple[int, ...]):
    return storage((1 << (qubits - 1 - wire) for wire in wires), dtype="int64")


def basis_state(qubits: int, *, dtype: str = "complex128"):
    if dtype not in ("complex64", "complex128"): raise ValueError("量子振幅只支持 complex64 或 complex128")
    result = CUDAStorage(2 ** qubits, dtype)
    _launch("q_basis", result.size, _p(result), _i(result.size), dtype=dtype, source="quantum.cu")
    return result


def apply_gate(amplitudes, qubits: int, wires: tuple[int, ...], matrix: tuple[tuple[complex, ...], ...]):
    masks = _quantum_masks(qubits, wires)
    coefficients = storage((value for row in matrix for value in row), dtype=amplitudes.dtype)
    selected = 0
    for wire in wires: selected |= 1 << (qubits - 1 - wire)
    result = CUDAStorage(amplitudes.size, amplitudes.dtype)
    _launch("q_gate", result.size, _p(amplitudes), _p(coefficients), _p(masks), _p(result),
            _i(result.size), _i(len(wires)), _i(selected), dtype=result.dtype, source="quantum.cu")
    return result


def probabilities(amplitudes, qubits: int, wires: tuple[int, ...]):
    masks = _quantum_masks(qubits, wires)
    hidden = tuple(wire for wire in range(qubits) if wire not in wires)
    hidden_masks = _quantum_masks(qubits, hidden)
    result = CUDAStorage(2 ** len(wires), "float32" if amplitudes.dtype == "complex64" else "float64")
    _launch("q_probabilities", result.size, _p(amplitudes), _p(masks), _p(hidden_masks), _p(result),
            _i(result.size), _i(len(wires)), _i(len(hidden)), dtype=result.dtype, source="quantum.cu")
    return result


def collapse(amplitudes, qubits: int, wires: tuple[int, ...], outcome: int, probability: float):
    masks = _quantum_masks(qubits, wires)
    result = CUDAStorage(amplitudes.size, amplitudes.dtype)
    _launch("q_collapse", result.size, _p(amplitudes), _p(masks), _p(result), _i(result.size),
            _i(len(wires)), _i(outcome), _s(probability ** -0.5, result.dtype),
            dtype=result.dtype, source="quantum.cu")
    return result


def inner(left, right) -> complex:
    # This is the explicit scalar readout used by State normalization/inspection.
    if left.dtype in ("float32", "float64"): return complex(real_inner(left, right).tolist()[0])
    if left.dtype not in ("complex64", "complex128"): raise ValueError("CUDA 内积只支持实数或复数浮点存储")
    result = CUDAStorage(1, left.dtype)
    _launch("q_inner", 1, _p(left), _p(right), _p(result), _i(left.size),
            dtype=left.dtype, source="quantum.cu")
    return result.tolist()[0]


def real_inner(left, right, scale: float = 1.0):
    if left.dtype not in ("float32", "float64", "complex64", "complex128"):
        raise ValueError("量子内积只支持 float32/64 或 complex64/128")
    if (left.device, left.dtype, left.size) != (right.device, right.dtype, right.size):
        raise ValueError("内积存储的设备、精度与元素数量必须相同")
    dtype = "float32" if left.dtype in ("float32", "complex64") else "float64"
    result = CUDAStorage(1, dtype)
    components = left.size * (2 if left.dtype.startswith("complex") else 1)
    _launch("q_real_inner", 1, _p(left), _p(right), _p(result), _i(components), _s(scale, dtype),
            dtype=dtype, source="quantum.cu")
    return result


def weighted_sum(values: tuple, weights):
    result = full(values[0].size, dtype=values[0].dtype)
    components = result.size * (2 if result.dtype.startswith("complex") else 1)
    for index, data in enumerate(values):
        _launch("q_weighted_sum", components, _p(data), _p(weights), _p(result), _i(components),
                _i(index), dtype=result.dtype, source="quantum.cu")
    return result


def probabilities_backward(gradient, amplitudes, qubits: int, wires: tuple[int, ...]):
    masks = _quantum_masks(qubits, wires)
    result = CUDAStorage(amplitudes.size, amplitudes.dtype)
    _launch("q_probabilities_backward", result.size, _p(gradient), _p(amplitudes), _p(masks),
            _p(result), _i(result.size), _i(len(wires)), dtype=result.dtype, source="quantum.cu")
    return result


def apply_rotation(amplitudes, qubits: int, wire: int, name: str, angle, *, adjoint: bool = False, derivative: bool = False):
    dtype = "float32" if amplitudes.dtype == "complex64" else "float64"
    if (angle.device, angle.dtype, angle.size) != (amplitudes.device, dtype, 1):
        raise ValueError("旋转角度必须是与量子态同设备、同精度的单元素存储")
    result = CUDAStorage(amplitudes.size, amplitudes.dtype)
    _launch("q_rotation", result.size, _p(amplitudes), _p(angle), _p(result), _i(result.size),
            _i(1 << (qubits - 1 - wire)), _i({"RX": 0, "RY": 1, "RZ": 2}[name]),
            _i(adjoint), _i(derivative), dtype=result.dtype, source="quantum.cu")
    return result
