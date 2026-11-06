from math import prod
from itertools import product, zip_longest
from .types import Data, Index, IntegerData, Nested, Shape


def infer_shape(data: Data) -> Shape:
    """推断数据形状，并检查嵌套结构是否规整。"""
    if isinstance(data, (int, float)): return ()
    if not isinstance(data, (list, tuple)): raise TypeError("输入必须是数字、列表或元组")
    if not data: return (0,)

    shapes = [infer_shape(item) for item in data]
    child_shape = shapes[0]
    if any(shape != child_shape for shape in shapes): raise ValueError("数据形状不一致，同一层的各项必须具有相同形状")
    return len(data), *child_shape


def flatten(data: Data) -> list[float]:
    """按原顺序，将数据展开为一维浮点数列表。"""
    if isinstance(data, (int, float)): return [float(data)]
    return [value for item in data for value in flatten(item)]


def flatten_indices(data: IntegerData) -> list[int]:
    """展开整数编号，保持原始整数，拒绝浮点数和 Tensor。"""
    if type(data) is int: return [data]
    if not isinstance(data, (list, tuple)): raise TypeError("编号必须是整数、列表或元组，不能使用浮点数或 Tensor")
    return [index for item in data for index in flatten_indices(item)]


def unflatten[T](values: list[T], shape: Shape) -> Nested[T]:
    """根据形状还原嵌套列表，保留元素的原始类型。"""
    if any(size < 0 for size in shape): raise ValueError("形状中的维度长度不能为负数")
    if len(values) != prod(shape): raise ValueError("元素数量与目标形状不匹配")
    if not shape: return values[0]

    block_size = prod(shape[1:])
    return [unflatten(values[i * block_size:(i + 1) * block_size], shape[1:],) for i in range(shape[0])]


def broadcast_shape(left: Shape, right: Shape) -> Shape:
    """根据两个输入形状，推断广播后的结果形状。"""
    result: list[int] = []
    for a, b in zip_longest(reversed(left), reversed(right), fillvalue=1):
        if a == b: result.append(a)
        elif a == 1: result.append(b)
        elif b == 1: result.append(a)
        else: raise ValueError(f"形状 {left} 和 {right} 无法广播")
    return tuple(reversed(result))


def strides(shape: Shape) -> tuple[int, ...]:
    """计算连续存储中，各维度前进一格对应的一维步长。"""
    return tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))


def index_positions(shape: Shape, index: Index) -> tuple[Shape, list[int]]:
    """将整数、切片、... 和新增轴转换成结果形状与原数据的位置。"""
    selectors = index if isinstance(index, tuple) else (index,)
    consumed = sum(part is not None and part is not Ellipsis for part in selectors)
    ellipses = sum(part is Ellipsis for part in selectors)
    if ellipses > 1: raise IndexError("索引中最多使用一个省略号")
    if consumed > len(shape): raise IndexError("索引的轴数超过张量维数")
    expanded = []
    for part in selectors:
        expanded.extend([slice(None)] * (len(shape) - consumed) if part is Ellipsis else [part])
    if not ellipses: expanded.extend([slice(None)] * (len(shape) - consumed))

    target, choices = [], []
    steps = strides(shape)
    axis = 0
    for part in expanded:
        if part is None:
            target.append(1)
            continue
        length, step = shape[axis], steps[axis]
        if isinstance(part, int):
            position = part + length if part < 0 else part
            if not 0 <= position < length: raise IndexError(f"第 {axis} 轴的索引 {part} 超出范围")
            choices.append([position * step])
        elif isinstance(part, slice):
            if part.step == 0: raise ValueError("切片步长不能为零")
            positions = range(*part.indices(length))
            target.append(len(positions))
            choices.append([position * step for position in positions])
        else:
            raise TypeError("索引只支持整数、切片、省略号和 None")
        axis += 1
    return tuple(target), [sum(items) for items in product(*choices)]


def normalize_axis(axis: int, ndim: int) -> int:
    """检查轴是否有效，并将负轴转换为非负轴。"""
    if not -ndim <= axis < ndim: raise ValueError(f"axis={axis} 超出 {ndim} 维数据的范围")
    return axis % ndim


def reduced_shape(shape: Shape, axis: int | None = None, keepdims: bool = False) -> Shape:
    """计算归约后的形状，可保留被归约的维度。"""
    if axis is None: return (1,) * len(shape) if keepdims else ()
    axis = normalize_axis(axis, len(shape))
    if keepdims: return shape[:axis] + (1,) + shape[axis + 1:]
    return shape[:axis] + shape[axis + 1:]


def reshape_shape(size: int, shape: Shape) -> Shape:
    """检查目标形状，并推断其中的一个 -1。"""
    if any(length < -1 for length in shape): raise ValueError("目标维度只能是非负整数或 -1")
    if shape.count(-1) > 1: raise ValueError("目标形状最多包含一个 -1")
    if -1 in shape:
        known_size = prod(length for length in shape if length != -1)
        if known_size == 0: raise ValueError("已知维度的乘积为零，无法推断 -1")
        if size % known_size != 0: raise ValueError("元素数量无法被已知维度的乘积整除")
        inferred = size // known_size
        shape = tuple(inferred if length == -1 else length for length in shape)
    if prod(shape) != size: raise ValueError(f"目标形状 {shape} 与元素数量 {size} 不匹配")
    return shape


def transpose_axes(ndim: int, axes: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """确定转置的轴顺序，要求每个轴恰好出现一次。"""
    if axes is None: return tuple(reversed(range(ndim)))
    axes = tuple(normalize_axis(axis, ndim) for axis in axes)
    if len(axes) != ndim or len(set(axes)) != ndim: raise ValueError("转置的轴顺序必须包含全部维度，且不能重复")
    return axes


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """将一个整数写成相同的高、宽，二元组则保留原值。"""
    result = (value, value) if isinstance(value, int) else value
    if len(result) != 2: raise ValueError("高和宽必须用一个整数或两个整数表示")
    return result


def window_shape(shape: Shape, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> Shape:
    """计算 NCHW 数据经过滑动窗口后的形状，通道数保持不变。"""
    if len(shape) != 4: raise ValueError("图片输入必须为（批次，通道，高，宽）四维数据")
    if any(size <= 0 for size in shape[1:]): raise ValueError("图片的通道数、高和宽必须大于零")
    if any(size <= 0 for size in (*kernel_size, *stride)): raise ValueError("窗口大小和步长必须大于零")
    if any(size < 0 for size in padding): raise ValueError("填充大小不能为负数")
    height, width = ((size + 2 * pad - kernel) // step + 1 for size, kernel, step, pad in zip(shape[2:], kernel_size, stride, padding, strict=True))
    if height <= 0 or width <= 0: raise ValueError("窗口超过填充后的图片大小")
    return shape[:2] + (height, width)
