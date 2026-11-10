from functools import partial
from types import SimpleNamespace

from ...backends import cuda_lm
from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import flatten_indices, index_layout, layout_positions, infer_shape, normalize_axis, reshape_shape, transpose_axes
from ..types import Computation, Index, IntegerData, Operand, Shape


def embedding(indices: IntegerData, weight: Operand) -> Computation:
    """用整数编号查权重表，保留编号的形状并在末尾增加特征轴。"""
    shape = infer_shape(indices)
    identifiers = flatten_indices(indices)
    if weight.ndim != 2 or any(size <= 0 for size in weight.shape): raise ValueError("嵌入权重必须为非空的二维表格（编号数量，特征数）")
    count, features = weight.shape
    if any(index < 0 or index >= count for index in identifiers): raise IndexError("嵌入编号超出权重表范围")
    if weight.storage.device == 'cuda' and weight.storage.dtype in ('float32', 'float64'):
        values = cuda_lm.embedding(weight.storage, identifiers, features)
        return values, (*shape, features), partial(_embedding_backward, identifiers=identifiers, count=count, features=features)
    positions = [index * features + feature for index in identifiers for feature in range(features)]
    backend = get_backend(weight.storage.device)
    values = backend.gather(weight.storage, positions)
    return values, (*shape, features), partial(_select_backward, positions=positions, size=weight.size)


def select(data: Operand, index: Index) -> Computation:
    """按索引读取数据；反向时将梯度加回原来的位置。"""
    shape, steps, offset = index_layout(data.shape, index)
    backend = get_backend(data.storage.device)
    values = (backend.gather_slice(data.storage, shape, steps, offset) if hasattr(backend, 'gather_slice') else
              backend.gather(data.storage, layout_positions(shape, steps, offset)))
    return values, shape, partial(_slice_backward, shape=shape, steps=steps, offset=offset, size=data.size)


def _slice_backward(gradient, shape, steps, offset, size):
    backend = get_backend(gradient.device)
    values = (backend.scatter_slice(gradient, shape, steps, offset, size) if hasattr(backend, 'scatter_slice') else
              backend.scatter(gradient, layout_positions(shape, steps, offset), size))
    return (values,)


def _embedding_backward(gradient, identifiers, count, features):
    return (cuda_lm.embedding_backward(gradient, identifiers, count, features),)


def concatenate(data: tuple[Operand, ...], axis: int = 0) -> Computation:
    if not data: raise ValueError('concatenate 至少需要一个张量')
    axis = normalize_axis(axis, data[0].ndim)
    target = list(data[0].shape)
    if any(item.ndim != len(target) or any(a != b for i, (a, b) in enumerate(zip(item.shape, target)) if i != axis) for item in data):
        raise ValueError('concatenate 除拼接轴以外的形状必须相同')
    target[axis] = sum(item.shape[axis] for item in data)
    backend = get_backend(*(item.storage for item in data))
    if any(item.storage.dtype != data[0].storage.dtype for item in data): raise ValueError('concatenate 精度必须相同')
    values = backend.concatenate(tuple(item.storage for item in data), tuple(item.shape for item in data), axis)
    return values, tuple(target), partial(_concatenate_backward, shape=tuple(target), sizes=tuple(item.shape[axis] for item in data), axis=axis)


def _concatenate_backward(gradient, shape, sizes, axis):
    source = SimpleNamespace(storage=gradient, shape=shape, size=gradient.size)
    result, start = [], 0
    for size in sizes:
        index = [slice(None)] * len(shape)
        index[axis] = slice(start, start + size)
        result.append(select(source, tuple(index))[0])
        start += size
    return tuple(result)


def _select_backward(gradient: Storage, positions: list[int], size: int) -> tuple[Storage, ...]:
    """被选位置接收输出梯度，未选位置为零。"""
    backend = get_backend(gradient.device)
    return (backend.scatter(gradient, positions, size),)


def stack(data: tuple[Operand, ...], axis: int = 0) -> Computation:
    """沿新增的轴组合相同形状的数据，反向时分别取回各项梯度。"""
    if not data: raise ValueError("stack 至少需要一项数据")
    shape = data[0].shape
    if any(item.shape != shape for item in data): raise ValueError("stack 的各项必须具有相同形状")
    axis = normalize_axis(axis, len(shape) + 1)
    axes = (*range(1, axis + 1), 0, *range(axis + 1, len(shape) + 1))
    target = shape[:axis] + (len(data),) + shape[axis:]
    backend = get_backend(*(item.storage for item in data))
    values = backend.concatenate(tuple(item.storage for item in data), tuple((item.size,) for item in data))
    if axis: values = backend.transpose(values, (len(data), *shape), axes=axes)
    return values, target, partial(_stack_backward, shape=target, axes=axes, size=data[0].size, count=len(data))


def _stack_backward(gradient: Storage, shape: Shape, axes: tuple[int, ...], size: int, count: int) -> tuple[Storage, ...]:
    """将新增轴移回最前面，再按每个输入的元素数量拆分。"""
    inverse = tuple(axes.index(axis) for axis in range(len(axes)))
    backend = get_backend(gradient.device)
    values = backend.transpose(gradient, shape, axes=inverse)
    return tuple(backend.gather(values, range(index * size, (index + 1) * size)) for index in range(count))


def transpose(data: Operand, axes: tuple[int, ...] | None = None) -> Computation:
    """按照指定轴顺序转置，默认反转全部轴。"""
    axes = transpose_axes(len(data.shape), axes)
    target_shape = tuple(data.shape[axis] for axis in axes)
    backend = get_backend(data.storage.device)
    values = backend.transpose(data.storage, data.shape, axes=axes)
    return values, target_shape, partial(_transpose_backward, shape=target_shape, axes=axes)


def _transpose_backward(gradient: Storage, shape: Shape, axes: tuple[int, ...]) -> tuple[Storage, ...]:
    """按前向轴顺序的逆置换，将梯度转回输入排列。"""
    inverse = tuple(axes.index(axis) for axis in range(len(axes)))
    backend = get_backend(gradient.device)
    return (backend.transpose(gradient, shape, axes=inverse),)


def reshape(data: Operand, shape: Shape) -> Computation:
    """保持元素顺序重新组织数据，支持一个 -1。"""
    target_shape = reshape_shape(data.size, shape)
    return data.storage.copy(), target_shape, _reshape_backward


def _reshape_backward(gradient: Storage) -> tuple[Storage, ...]:
    """一维顺序不变，传播引擎使用父节点的原形状保存梯度。"""
    return (gradient,)
