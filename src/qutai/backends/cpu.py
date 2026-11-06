from array import array
from collections.abc import Callable, Iterable, Iterator
import operator
from functools import lru_cache
from itertools import product
from math import erf, exp as scalar_exp, fsum, hypot, log as scalar_log, pi, prod, sqrt

from ..core.shape import broadcast_shape, normalize_axis, reduced_shape, strides, transpose_axes, window_shape
from ..core.types import Shape
from .storage import Storage


class CPUStorage(Storage):
    """连续实数缓冲区；复数按实部、虚部交错保存。"""
    device = "cpu"

    def __init__(self, values: Iterable[float | complex] = (), *, dtype: str = "float64") -> None:
        if dtype not in ("float32", "float64", "complex64", "complex128", "int64"): raise ValueError(f"CPU 尚不支持精度：{dtype}")
        self.dtype = dtype
        if dtype.startswith("complex"): values = (part for value in values for part in (complex(value).real, complex(value).imag))
        self.buffer = array("q" if dtype == "int64" else "f" if dtype in ("float32", "complex64") else "d", values)

    @property
    def size(self) -> int:
        return len(self.buffer) // (2 if self.dtype.startswith("complex") else 1)

    @property
    def nbytes(self) -> int:
        return len(self.buffer) * self.buffer.itemsize

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self.buffer) if not self.dtype.startswith("complex") else (complex(self.buffer[i], self.buffer[i + 1]) for i in range(0, len(self.buffer), 2))

    def __getitem__(self, index):
        if isinstance(index, slice):
            if self.dtype.startswith("complex"): return CPUStorage((self[i] for i in range(*index.indices(self.size))), dtype=self.dtype)
            result = CPUStorage(dtype=self.dtype)
            result.buffer = self.buffer[index]
            return result
        if not self.dtype.startswith("complex"): return self.buffer[index]
        if index < 0: index += self.size
        if not 0 <= index < self.size: raise IndexError("存储索引超出范围")
        return complex(self.buffer[2 * index], self.buffer[2 * index + 1])

    def __setitem__(self, index, value) -> None:
        if not self.dtype.startswith("complex"):
            self.buffer[index] = array(self.buffer.typecode, value) if isinstance(index, slice) else value
        elif isinstance(index, slice):
            values = self.tolist()
            values[index] = list(value)
            self.buffer = CPUStorage(values, dtype=self.dtype).buffer
        else:
            if index < 0: index += self.size
            if not 0 <= index < self.size: raise IndexError("存储索引超出范围")
            self.buffer[2 * index], self.buffer[2 * index + 1] = complex(value).real, complex(value).imag

    def copy(self) -> CPUStorage:
        result = CPUStorage(dtype=self.dtype)
        result.buffer = self.buffer[:]
        return result

    def tolist(self) -> list[float] | list[complex]:
        return list(self)

    def __eq__(self, other) -> bool:
        return self.tolist() == (other.tolist() if isinstance(other, Storage) else other)

    def __repr__(self) -> str:
        return repr(self.tolist())


def storage(values: Iterable[float | complex] = (), *, dtype: str = "float64") -> CPUStorage:
    """导入独立的 CPU 数值存储。"""
    if isinstance(values, CPUStorage) and values.dtype == dtype: return values.copy()
    return CPUStorage(values, dtype=dtype)


def full(size: int, value: float | complex = 0.0, *, dtype: str = "float64") -> CPUStorage:
    """创建固定数量的同值元素。"""
    return CPUStorage((int(value) if dtype == "int64" else value for _ in range(size)), dtype=dtype)


def apply_gate(amplitudes: Storage, qubits: int, wires: tuple[int, ...], matrix: tuple[tuple[complex, ...], ...]) -> CPUStorage:
    """按所选比特分组，只用门的小矩阵更新各组复数振幅。"""
    offsets, selected = [0], 0
    for wire in wires:
        mask = 1 << (qubits - 1 - wire)
        selected |= mask
        offsets = [offset | bit for offset in offsets for bit in (0, mask)]
    result = storage(amplitudes, dtype=amplitudes.dtype)
    for base in range(len(amplitudes)):
        if base & selected: continue
        block = [amplitudes[base | offset] for offset in offsets]
        for row, offset in zip(matrix, offsets, strict=True):
            value = 0j
            for coefficient, amplitude in zip(row, block, strict=True): value += coefficient * amplitude
            result[base | offset] = value
    return result


def gather(values: Storage, positions: list[int]) -> CPUStorage:
    """按一维位置读取数据。"""
    return storage((values[position] for position in positions), dtype=getattr(values, "dtype", "float64"))


def scatter(values: Storage, positions: list[int], size: int) -> CPUStorage:
    """将梯度累加回原位置，未被选中的位置保持零。"""
    result = full(size, dtype=getattr(values, "dtype", "float64"))
    for position, value in zip(positions, values, strict=True):
        result[position] += value
    return result


def unary(values: Storage, operation: Callable[[float], float]) -> CPUStorage:
    """对每个元素应用一个运算。"""
    return storage((operation(value) for value in values), dtype=getattr(values, "dtype", "float64"))


def binary(left: Storage, right: Storage, operation: Callable[[float, float], float]) -> CPUStorage:
    """对两个等长列表的对应元素应用一个运算。"""
    return storage((operation(a, b) for a, b in zip(left, right, strict=True)), dtype=getattr(left, "dtype", "float64"))


def broadcast(values: Storage, shape: Shape, target: Shape) -> CPUStorage:
    """将一维数据按合法的广播目标形状展开。"""
    padded_shape = (1,) * (len(target) - len(shape)) + shape
    steps = strides(padded_shape)
    result: list[float] = []
    for coordinate in product(*(range(size) for size in target)):
        offset = 0
        for position, length, step in zip(coordinate, padded_shape, steps):
            if length != 1: offset += position * step
        result.append(values[offset])
    return storage(result, dtype=getattr(values, "dtype", "float64"))


def matmul(left: Storage, right: Storage, left_shape: Shape, right_shape: Shape,) -> CPUStorage:
    """向量临时升为矩阵，按广播后的批次坐标读取原数据并相乘。"""
    if len(left_shape) == 1: left_shape = (1, *left_shape)
    if len(right_shape) == 1: right_shape = (*right_shape, 1)
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    left_shape = (1,) * (len(batches) + 2 - len(left_shape)) + left_shape
    right_shape = (1,) * (len(batches) + 2 - len(right_shape)) + right_shape
    left_steps, right_steps = strides(left_shape), strides(right_shape)
    rows, inner = left_shape[-2:]
    columns = right_shape[-1]
    result = full(prod(batches) * rows * columns, dtype=getattr(left, "dtype", "float64"))

    for batch, coordinate in enumerate(product(*(range(size) for size in batches))):
        left_start = right_start = 0
        for axis, position in enumerate(coordinate):
            if left_shape[axis] != 1: left_start += position * left_steps[axis]
            if right_shape[axis] != 1: right_start += position * right_steps[axis]
        for row in range(rows):
            for column in range(columns):
                total = 0.0
                for k in range(inner):
                    total += left[left_start + row * inner + k] * right[right_start + k * columns + column]
                result[(batch * rows + row) * columns + column] = total
    return result


def transpose(values: Storage, shape: Shape, axes: tuple[int, ...] | None = None) -> CPUStorage:
    """按照指定轴顺序重新排列数据，默认反转全部轴。"""
    axes = transpose_axes(len(shape), axes)
    target_shape = tuple(shape[axis] for axis in axes)
    target_steps = strides(target_shape)
    result = full(len(values), dtype=getattr(values, "dtype", "float64"))

    coordinates = product(*(range(size) for size in shape))
    for value, coordinate in zip(values, coordinates, strict=True):
        target_coordinate = tuple(coordinate[axis] for axis in axes)
        offset = 0
        for position, step in zip(target_coordinate, target_steps):
            offset += position * step
        result[offset] = value
    return result


def sum(values: Storage, shape: Shape | None = None, axis: int | None = None,) -> CPUStorage:
    """全局求和，或沿指定轴求和。"""
    if axis is None:
        total = 0.0
        for value in values:
            total += value
        return storage([total], dtype=getattr(values, "dtype", "float64"))
    if shape is None: raise ValueError("按轴求和时必须提供数据形状")

    axis = normalize_axis(axis, len(shape))
    target_shape = reduced_shape(shape, axis)
    target_steps = strides(target_shape)
    result = full(prod(target_shape), dtype=getattr(values, "dtype", "float64"))
    coordinates = product(*(range(size) for size in shape))
    for value, coordinate in zip(values, coordinates, strict=True):
        target_coordinate = coordinate[:axis] + coordinate[axis + 1:]
        offset = 0
        for position, step in zip(target_coordinate, target_steps):
            offset += position * step
        result[offset] += value
    return result


def mean(values: Storage, shape: Shape | None = None, axis: int | None = None) -> CPUStorage:
    """计算全局或指定轴的平均值，归约元素数量须大于零。"""
    if axis is None:
        count = len(values)
    else:
        if shape is None: raise ValueError("按轴求平均值时必须提供数据形状")
        axis = normalize_axis(axis, len(shape))
        count = shape[axis]
    totals = sum(values, shape, axis=axis)
    return storage((total / count for total in totals), dtype=getattr(values, "dtype", "float64"))


def unbroadcast(values: Storage, shape: Shape, target: Shape) -> CPUStorage:
    """将广播后各位置的梯度求和，还原到原输入形状。"""
    padded = (1,) * (len(shape) - len(target)) + target
    current_shape = shape

    for axis, (length, original) in enumerate(zip(shape, padded, strict=True)):
        if original == 1 and length != 1:
            values = sum(values, current_shape, axis=axis)
            current_shape = current_shape[:axis] + (1,) + current_shape[axis + 1:]

    return values if isinstance(values, Storage) else storage(values)


def _axis_indices(shape: Shape, axis: int) -> Iterator[range]:
    """依次给出沿指定轴计算的各组元素在一维存储中的位置。"""
    axis = normalize_axis(axis, len(shape))
    step = prod(shape[axis + 1:])
    width = shape[axis] * step
    return (range(outer * width + offset, (outer + 1) * width + offset, step)
            for outer in range(prod(shape[:axis])) for offset in range(step))


def argmax(values: Storage, shape: Shape, axis: int | None = None) -> list[int]:
    """返回各组最大值的第一个编号；不指定轴时按展开顺序编号。"""
    if axis is None: return [max(range(len(values)), key=values.__getitem__)]
    result = []
    for indices in _axis_indices(shape, axis):
        items = [values[index] for index in indices]
        result.append(max(range(len(items)), key=items.__getitem__))
    return result


def softmax(values: Storage, shape: Shape, axis: int) -> CPUStorage:
    """每组减去最大值后计算 Softmax，避免指数溢出。"""
    result = full(len(values), dtype=getattr(values, "dtype", "float64"))
    for indices in _axis_indices(shape, axis):
        items = [values[index] for index in indices]
        peak = max(items)
        weights = [scalar_exp(value - peak) for value in items]
        total = fsum(weights)
        for index, weight in zip(indices, weights, strict=True):
            result[index] = weight / total
    return result


def cross_entropy(values: Storage, shape: Shape, labels: tuple[int, ...]) -> CPUStorage:
    """由二维原始得分计算平均交叉熵，保留极小概率对应的有限损失。"""
    rows, columns = shape
    losses = []
    for row, label in enumerate(labels):
        scores = values[row * columns:(row + 1) * columns]
        peak = max(scores)
        shifted = [score - peak for score in scores]
        losses.append(scalar_log(fsum(scalar_exp(score) for score in shifted)) - shifted[label])
    return storage([fsum(loss / rows for loss in losses)], dtype=getattr(values, "dtype", "float64"))


@lru_cache(maxsize=16)
def _windows(shape: Shape, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """预先列出每个窗口中（卷积核位置，图片位置），越过边界的零填充不参与计算。"""
    channels, height, width = shape
    kernel_height, kernel_width = kernel_size
    _, _, rows, columns = window_shape((1, *shape), kernel_size, stride, padding)
    windows = []
    for row in range(rows):
        for column in range(columns):
            positions = []
            for channel in range(channels):
                for y in range(kernel_height):
                    source_y = row * stride[0] + y - padding[0]
                    if not 0 <= source_y < height: continue
                    for x in range(kernel_width):
                        source_x = column * stride[1] + x - padding[1]
                        if 0 <= source_x < width:
                            kernel_index = (channel * kernel_height + y) * kernel_width + x
                            source_index = (channel * height + source_y) * width + source_x
                            positions.append((kernel_index, source_index))
            windows.append(tuple(positions))
    return tuple(windows)


def conv2d(values: Storage, weight: Storage, bias: Storage | None, shape: Shape, weight_shape: Shape, stride: tuple[int, int], padding: tuple[int, int]) -> CPUStorage:
    """按 NCHW 输入、OIHW 卷积核计算二维卷积；卷积核不翻转。"""
    windows = _windows(shape[1:], (weight_shape[2], weight_shape[3]), stride, padding)
    image_size, kernel_size = prod(shape[1:]), prod(weight_shape[1:])
    result = []
    for batch in range(shape[0]):
        image = values[batch * image_size:(batch + 1) * image_size]
        for channel in range(weight_shape[0]):
            kernel = weight[channel * kernel_size:(channel + 1) * kernel_size]
            initial = bias[channel] if bias is not None else 0.0
            for window in windows:
                total = initial
                for kernel_index, source_index in window:
                    total += image[source_index] * kernel[kernel_index]
                result.append(total)
    return storage(result, dtype=getattr(values, "dtype", "float64"))


def conv2d_backward(gradient: Storage, values: Storage, weight: Storage, shape: Shape, weight_shape: Shape, stride: tuple[int, int], padding: tuple[int, int], *, input_grad: bool = True) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """把每个输出梯度乘回输入和卷积核；重叠窗口的贡献相加。"""
    windows = _windows(shape[1:], (weight_shape[2], weight_shape[3]), stride, padding)
    image_size, kernel_size = prod(shape[1:]), prod(weight_shape[1:])
    input_gradient, weight_gradient, bias_gradient = [], full(len(weight), dtype=getattr(gradient, "dtype", "float64")), full(weight_shape[0], dtype=getattr(gradient, "dtype", "float64"))
    offset = 0
    for batch in range(shape[0]):
        image = values[batch * image_size:(batch + 1) * image_size]
        image_gradient = full(image_size, dtype=getattr(gradient, "dtype", "float64"))
        for channel in range(weight_shape[0]):
            start = channel * kernel_size
            kernel = weight[start:start + kernel_size]
            kernel_gradient = weight_gradient[start:start + kernel_size]
            for window in windows:
                incoming = gradient[offset]
                offset += 1
                bias_gradient[channel] += incoming
                if incoming == 0: continue
                for kernel_index, source_index in window:
                    kernel_gradient[kernel_index] += image[source_index] * incoming
                    if input_grad:
                        image_gradient[source_index] += kernel[kernel_index] * incoming
            weight_gradient[start:start + kernel_size] = kernel_gradient
        input_gradient.extend(image_gradient)
    return storage(input_gradient, dtype=getattr(gradient, "dtype", "float64")), weight_gradient, bias_gradient


def max_pool2d(values: Storage, shape: Shape, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> tuple[CPUStorage, CPUStorage]:
    """每个窗口保留最大值和原位置；填充不参加比较，并列时保留第一个。"""
    windows = _windows((1, *shape[2:]), kernel_size, stride, padding)
    plane_size = prod(shape[2:])
    result, indices = [], []
    for plane in range(shape[0] * shape[1]):
        start = plane * plane_size
        for window in windows:
            index = window[0][1]
            for _, candidate in window[1:]:
                if values[start + candidate] > values[start + index]: index = candidate
            result.append(values[start + index])
            indices.append(start + index)
    return storage(result, dtype=getattr(values, "dtype", "float64")), storage(indices, dtype="int64")


def max_pool2d_backward(gradient: Storage, indices: Storage, size: int) -> CPUStorage:
    """梯度回到各窗口选中的位置，同一位置被多次选中时累加。"""
    result = full(size, dtype=getattr(gradient, "dtype", "float64"))
    for index, incoming in zip(indices, gradient, strict=True):
        result[index] += incoming
    return result


def channel_stats(values: Storage, shape: Shape) -> tuple[CPUStorage, CPUStorage]:
    """NCHW 按通道跨批次与空间统计均值、总体方差。"""
    batch, channels, height, width = shape
    area = height * width
    count = batch * area
    averages, variances = full(channels, dtype=getattr(values, "dtype", "float64")), full(channels, dtype=getattr(values, "dtype", "float64"))
    for channel in range(channels):
        items = [values[(sample * channels + channel) * area + pixel] for sample in range(batch) for pixel in range(area)]
        average = fsum(items) / count
        averages[channel] = average
        variances[channel] = fsum((value - average) ** 2 for value in items) / count
    return averages, variances


def batch_norm2d(values: Storage, weight: Storage, bias: Storage, shape: Shape, averages: Storage, variances: Storage, eps: float) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """按通道归一化并缩放平移，同时返回本次前向的归一化值和逆标准差。"""
    channels, height, width = shape[1:]
    area = height * width
    inverse = [(variance + eps) ** -0.5 for variance in variances]
    normalized, result = full(len(values), dtype=getattr(values, "dtype", "float64")), full(len(values), dtype=getattr(values, "dtype", "float64"))
    for index, value in enumerate(values):
        channel = index // area % channels
        normalized[index] = (value - averages[channel]) * inverse[channel]
        result[index] = normalized[index] * weight[channel] + bias[channel]
    return result, normalized, storage(inverse, dtype=getattr(values, "dtype", "float64"))


def batch_norm2d_backward(gradient: Storage, normalized: Storage, inverse: Storage, weight: Storage, shape: Shape, train: bool) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """计算输入、缩放参数、平移参数的梯度；训练时包含均值和方差的导数。"""
    batch, channels, height, width = shape
    area, count = height * width, batch * height * width
    input_gradient = full(len(gradient), dtype=getattr(gradient, "dtype", "float64"))
    weight_gradient, bias_gradient = full(channels, dtype=getattr(gradient, "dtype", "float64")), full(channels, dtype=getattr(gradient, "dtype", "float64"))
    for index, incoming in enumerate(gradient):
        channel = index // area % channels
        weight_gradient[channel] += incoming * normalized[index]
        bias_gradient[channel] += incoming
    for index, incoming in enumerate(gradient):
        channel = index // area % channels
        if train:
            incoming -= (bias_gradient[channel] + normalized[index] * weight_gradient[channel]) / count
        input_gradient[index] = incoming * weight[channel] * inverse[channel]
    return input_gradient, weight_gradient, bias_gradient


def layer_norm(values: Storage, weight: Storage, bias: Storage, width: int, eps: float) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """每组最后一轴独立归一化，保存归一化值和各组的逆标准差。"""
    result, normalized, inverse = [], [], []
    for start in range(0, len(values), width):
        items = values[start:start + width]
        average = fsum(items) / width
        centered = [value - average for value in items]
        scale = (fsum(value * value for value in centered) / width + eps) ** -0.5
        row = [value * scale for value in centered]
        normalized.extend(row)
        inverse.append(scale)
        result.extend(value * gamma + beta for value, gamma, beta in zip(row, weight, bias, strict=True))
    return storage(result, dtype=getattr(values, "dtype", "float64")), storage(normalized, dtype=getattr(values, "dtype", "float64")), storage(inverse, dtype=getattr(values, "dtype", "float64"))


def layer_norm_backward(gradient: Storage, normalized: Storage, inverse: Storage, weight: Storage) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """各组独立传播输入梯度，缩放和平移参数的梯度跨组累加。"""
    width = len(weight)
    input_gradient = full(len(gradient), dtype=getattr(gradient, "dtype", "float64"))
    weight_gradient, bias_gradient = full(width, dtype=getattr(gradient, "dtype", "float64")), full(width, dtype=getattr(gradient, "dtype", "float64"))
    for group, scale in enumerate(inverse):
        start = group * width
        weighted = [gradient[start + index] * weight[index] for index in range(width)]
        average = fsum(weighted) / width
        projected = fsum(weighted[index] * normalized[start + index] for index in range(width)) / width
        for index in range(width):
            position = start + index
            input_gradient[position] = scale * (weighted[index] - average - normalized[position] * projected)
            weight_gradient[index] += gradient[position] * normalized[position]
            bias_gradient[index] += gradient[position]
    return input_gradient, weight_gradient, bias_gradient


def copy_into(target: CPUStorage, source: Storage) -> None:
    """覆盖数值但保留目标存储对象。"""
    if target.dtype != source.dtype or target.size != source.size: raise ValueError("复制存储的精度与元素数量必须相同")
    target.buffer[:] = source.buffer


def add(left: Storage, right: Storage) -> CPUStorage:
    return binary(left, right, operator.add)


def subtract(left: Storage, right: Storage) -> CPUStorage:
    return binary(left, right, operator.sub)


def multiply(left: Storage, right: Storage) -> CPUStorage:
    return binary(left, right, operator.mul)


def divide(left: Storage, right: Storage) -> CPUStorage:
    return binary(left, right, operator.truediv)


def negate(values: Storage) -> CPUStorage:
    return unary(values, operator.neg)


def square(values: Storage) -> CPUStorage:
    return unary(values, lambda value: value * value)


def scale(values: Storage, factor: float) -> CPUStorage:
    return unary(values, lambda value: value * factor)


def relu(values: Storage) -> CPUStorage:
    return unary(values, lambda value: max(value, 0.0))


def relu_backward(gradient: Storage, values: Storage) -> CPUStorage:
    return storage((incoming if value > 0 else 0.0 for incoming, value in zip(gradient, values, strict=True)), dtype=getattr(gradient, "dtype", "float64"))


def gelu(values: Storage) -> CPUStorage:
    return unary(values, lambda value: value * (1 + erf(value / sqrt(2))) / 2)


def gelu_backward(gradient: Storage, values: Storage) -> CPUStorage:
    return storage((incoming * ((1 + erf(value / sqrt(2))) / 2 + value * scalar_exp(-value * value / 2) / sqrt(2 * pi))
                    for incoming, value in zip(gradient, values, strict=True)), dtype=getattr(gradient, "dtype", "float64"))


def exp(values: Storage) -> CPUStorage:
    return unary(values, scalar_exp)


def log(values: Storage) -> CPUStorage:
    if any(value <= 0 for value in values): raise ValueError("自然对数的输入必须大于零")
    return unary(values, scalar_log)


def concatenate(values: tuple[Storage, ...], shapes: tuple[Shape, ...], axis: int = 0) -> CPUStorage:
    """沿指定轴拼接，逐个复制相同外层位置对应的连续区块。"""
    axis = normalize_axis(axis, len(shapes[0]))
    sizes = [prod(shape[axis:]) for shape in shapes]
    result = [value for outer in range(prod(shapes[0][:axis]))
              for data, size in zip(values, sizes, strict=True) for value in data[outer * size:(outer + 1) * size]]
    return storage(result, dtype=values[0].dtype)


def global_norm(values: tuple[Storage, ...]) -> float:
    """计算全部梯度的稳定 L2 范数。"""
    return hypot(*(value for data in values for value in data))


def cross_entropy_backward(gradient: Storage, scores: Storage, shape: Shape, labels: tuple[int, ...]) -> CPUStorage:
    """交叉熵的得分梯度，避免在上层逐项读取存储。"""
    rows, columns = shape
    result = softmax(scores, shape, axis=1)
    for row, label in enumerate(labels): result[row * columns + label] -= 1.0
    return scale(result, gradient[0] / rows)


def sgd_step(values: Storage, gradient: Storage, velocity: Storage | None, *, learning_rate: float, momentum: float, weight_decay: float) -> tuple[CPUStorage, CPUStorage | None]:
    """动量与权重衰减的纯数值更新。"""
    direction = add(gradient, scale(values, weight_decay)) if weight_decay else gradient
    updated = None
    if momentum:
        updated = add(scale(velocity, momentum), direction) if velocity is not None else storage(direction, dtype=getattr(values, "dtype", "float64"))
        direction = updated
    return subtract(values, scale(direction, learning_rate)), updated


def adamw_step(values: Storage, gradient: Storage, average: Storage, square_average: Storage, *, step: int, learning_rate: float, betas: tuple[float, float], eps: float, weight_decay: float) -> tuple[CPUStorage, CPUStorage, CPUStorage]:
    """偏差修正后的 Adam 更新；权重衰减独立作用于参数。"""
    beta1, beta2 = betas
    average = storage((beta1 * old + (1 - beta1) * value for old, value in zip(average, gradient, strict=True)), dtype=getattr(values, "dtype", "float64"))
    square_average = storage((beta2 * old + (1 - beta2) * value * value for old, value in zip(square_average, gradient, strict=True)), dtype=getattr(values, "dtype", "float64"))
    correction1, correction2 = 1 - beta1 ** step, 1 - beta2 ** step
    decay = 1 - learning_rate * weight_decay
    result = storage((value * decay - learning_rate * (m / correction1) / (sqrt(v / correction2) + eps)
                      for value, m, v in zip(values, average, square_average, strict=True)), dtype=getattr(values, "dtype", "float64"))
    return result, average, square_average


def _outcome(index: int, qubits: int, wires: tuple[int, ...]) -> int:
    """按调用者给出的比特顺序取出基态编号。"""
    result = 0
    for wire in wires: result = (result << 1) | ((index >> (qubits - 1 - wire)) & 1)
    return result


def basis_state(qubits: int, *, dtype: str = "complex128") -> CPUStorage:
    """创建全零计算基态。"""
    result = full(2 ** qubits, dtype=dtype)
    result[0] = 1
    return result


def probabilities(amplitudes: Storage, qubits: int, wires: tuple[int, ...]) -> CPUStorage:
    """将振幅概率累加到指定比特顺序的边缘分布。"""
    result = full(2 ** len(wires), dtype="float32" if amplitudes.dtype == "complex64" else "float64")
    for index, amplitude in enumerate(amplitudes): result[_outcome(index, qubits, wires)] += abs(amplitude) ** 2
    return result


def collapse(amplitudes: Storage, qubits: int, wires: tuple[int, ...], outcome: int, probability: float) -> CPUStorage:
    """保留指定测量结果并归一化，其余振幅归零。"""
    scale = sqrt(probability)
    return storage((value / scale if _outcome(index, qubits, wires) == outcome else 0j for index, value in enumerate(amplitudes)), dtype=amplitudes.dtype)


def inner(left: Storage, right: Storage) -> complex:
    """复内积；普通 State 数值读取和归一化检查的出口。"""
    result = 0j
    for a, b in zip(left, right, strict=True): result += a.conjugate() * b
    return result


def real_inner(left: Storage, right: Storage, scale: float = 1.0) -> CPUStorage:
    """返回实部内积的单元素存储，反向计算无需读取主机标量。"""
    return storage([scale * inner(left, right).real], dtype="float32" if left.dtype in ("float32", "complex64") else "float64")


def weighted_sum(values: tuple[Storage, ...], weights: Storage) -> CPUStorage:
    """将多个向量按同位置权重相加。"""
    result = full(values[0].size, dtype=values[0].dtype)
    for data, weight in zip(values, weights, strict=True):
        for index, value in enumerate(data): result[index] += weight * value
    return result


def probabilities_backward(gradient: Storage, amplitudes: Storage, qubits: int, wires: tuple[int, ...]) -> CPUStorage:
    """将边缘概率的上游梯度映射回完整振幅。"""
    return storage((gradient[_outcome(index, qubits, wires)] * value for index, value in enumerate(amplitudes)), dtype=amplitudes.dtype)


def is_available() -> bool:
    return True


def synchronize() -> None:
    """CPU 运算同步完成。"""
    pass


def from_storage(values: Storage, *, dtype: str) -> CPUStorage:
    """导入独立存储；设备间拷贝由目标后端实现。"""
    if isinstance(values, CPUStorage) and values.dtype == dtype: return values.copy()
    return storage(values.tolist(), dtype=dtype)
