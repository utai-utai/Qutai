from functools import partial

from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import normalize_axis, reduced_shape, unflatten
from ..types import Computation, Indices, Operand, Shape


def sum(data: Operand, axis: int | None = None, keepdims: bool = False) -> Computation:
    """计算全部元素或指定轴上的和，可保留归约维度。"""
    target_shape = reduced_shape(data.shape, axis, keepdims=keepdims)
    backend = get_backend(data.storage.device)
    values = backend.sum(data.storage, data.shape, axis=axis)
    return values, target_shape, partial(_sum_backward, shape=data.shape, axis=axis)


def _sum_backward(gradient: Storage, shape: Shape, axis: int | None) -> tuple[Storage, ...]:
    """补回归约维度，再将输出梯度广播到输入形状。"""
    kept_shape = reduced_shape(shape, axis, keepdims=True)
    backend = get_backend(gradient.device)
    return (backend.broadcast(gradient, kept_shape, shape),)


def mean(data: Operand, axis: int | None = None, keepdims: bool = False) -> Computation:
    """计算全部元素或指定轴上的平均值。"""
    target_shape = reduced_shape(data.shape, axis, keepdims=keepdims)
    count = data.size if axis is None else data.shape[axis]
    if count == 0: raise ValueError("归约范围内没有元素，无法求平均值")

    backend = get_backend(data.storage.device)
    values = backend.mean(data.storage, data.shape, axis=axis)
    return values, target_shape, partial(_mean_backward, shape=data.shape, axis=axis, count=count)


def _mean_backward(gradient: Storage, shape: Shape, axis: int | None, count: int) -> tuple[Storage, ...]:
    """恢复 sum 的输入梯度，再除以归约元素数量。"""
    values, = _sum_backward(gradient, shape, axis)
    backend = get_backend(gradient.device)
    return (backend.scale(values, 1 / count),)


def argmax(data: Operand, axis: int | None = None, keepdims: bool = False) -> Indices:
    """返回最大值的第一个整数编号；始终返回普通数据，不参与求导。"""
    if axis is not None: axis = normalize_axis(axis, len(data.shape))
    count = data.size if axis is None else data.shape[axis]
    if count == 0: raise ValueError("归约范围内没有元素，无法取最大值编号")
    shape = reduced_shape(data.shape, axis, keepdims)
    backend = get_backend(data.storage.device)
    return unflatten(backend.argmax(data.storage, data.shape, axis), shape)
