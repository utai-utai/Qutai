from functools import partial

from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import broadcast_shape
from ..types import Computation, Operand, Shape


def matmul(left: Operand, right: Operand) -> Computation:
    """计算向量、矩阵或批量矩阵乘积，仅在批次维度广播。"""
    if not left.shape or not right.shape: raise ValueError("矩阵乘法的输入至少需要一维")

    left_shape = (1, *left.shape) if len(left.shape) == 1 else left.shape
    right_shape = (*right.shape, 1) if len(right.shape) == 1 else right.shape
    if left_shape[-1] != right_shape[-2]: raise ValueError(f"矩阵形状 {left.shape} 和 {right.shape} 不兼容")
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    target_shape = batches + (() if len(left.shape) == 1 else (left_shape[-2],)) + (() if len(right.shape) == 1 else (right_shape[-1],))

    backend = get_backend(left.storage, right.storage)
    values = backend.matmul(left.storage, right.storage, left_shape, right_shape)
    return values, target_shape, partial(_matmul_backward, left=left, right=right, left_shape=left_shape, right_shape=right_shape)


def _matmul_backward(gradient: Storage, left: Operand, right: Operand, left_shape: Shape, right_shape: Shape) -> tuple[Storage, ...]:
    """只转置末两轴求矩阵梯度，再将广播批次上的梯度累加回输入。"""
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    shape = batches + (left_shape[-2], right_shape[-1])
    left_axes = (*range(len(left_shape) - 2), len(left_shape) - 1, len(left_shape) - 2)
    right_axes = (*range(len(right_shape) - 2), len(right_shape) - 1, len(right_shape) - 2)
    backend = get_backend(gradient.device)
    left_transposed = backend.transpose(left.storage, left_shape, axes=left_axes)
    right_transposed = backend.transpose(right.storage, right_shape, axes=right_axes)
    left_gradient = backend.matmul(gradient, right_transposed, shape, right_shape[:-2] + right_shape[-2:][::-1])
    right_gradient = backend.matmul(left_transposed, gradient, left_shape[:-2] + left_shape[-2:][::-1], shape)
    return (backend.unbroadcast(left_gradient, batches + left_shape[-2:], left_shape),
            backend.unbroadcast(right_gradient, batches + right_shape[-2:], right_shape))
