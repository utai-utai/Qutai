from functools import partial
from math import prod

from ...backends.device import get_backend
from ...backends.storage import Storage
from ...backends.precision import matmul_precision, set_matmul_precision
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
    return values, target_shape, partial(_matmul_backward, left=left, right=right, left_shape=left_shape, right_shape=right_shape, precision=matmul_precision())


def _matmul_backward(gradient: Storage, left: Operand, right: Operand, left_shape: Shape, right_shape: Shape, precision: str | None = None) -> tuple[Storage | None, ...]:
    with set_matmul_precision(precision):
        return _matmul_gradients(gradient, left, right, left_shape, right_shape)


def _matmul_gradients(gradient: Storage, left: Operand, right: Operand, left_shape: Shape, right_shape: Shape) -> tuple[Storage | None, ...]:
    """按末两轴的转置关系求梯度，仅计算需要的一侧并归约广播批次。"""
    batches = broadcast_shape(left_shape[:-2], right_shape[:-2])
    shape = batches + (left_shape[-2], right_shape[-1])
    left_axes = (*range(len(left_shape) - 2), len(left_shape) - 1, len(left_shape) - 2)
    right_axes = (*range(len(right_shape) - 2), len(right_shape) - 1, len(right_shape) - 2)
    backend = get_backend(gradient.device)
    left_gradient, right_gradient = None, None
    if hasattr(backend, "gemm") and left_shape[:-2] == right_shape[:-2]:
        rows, inner, columns = left_shape[-2], left_shape[-1], right_shape[-1]
        if left.requires_grad:
            left_gradient = backend.gemm(gradient, right.storage, rows, columns, inner,
                                         batch=prod(batches), transpose_right=True, out_dtype=left.storage.dtype)
        if right.requires_grad:
            right_gradient = backend.gemm(left.storage, gradient, inner, rows, columns,
                                          batch=prod(batches), transpose_left=True, out_dtype=right.storage.dtype)
    else:
        if left.requires_grad:
            transposed = backend.transpose(right.storage, right_shape, axes=right_axes)
            left_gradient = backend.matmul(gradient, transposed, shape, right_shape[:-2] + right_shape[-2:][::-1])
            left_gradient = backend.unbroadcast(left_gradient, batches + left_shape[-2:], left_shape)
        if right.requires_grad:
            transposed = backend.transpose(left.storage, left_shape, axes=left_axes)
            right_gradient = backend.matmul(transposed, gradient, left_shape[:-2] + left_shape[-2:][::-1], shape)
            right_gradient = backend.unbroadcast(right_gradient, batches + right_shape[-2:], right_shape)
    return left_gradient, right_gradient
