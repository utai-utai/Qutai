from functools import partial

from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import normalize_axis, reduced_shape
from ..types import Computation, Operand, Shape


def softmax(data: Operand, axis: int = -1) -> Computation:
    """沿指定轴将得分归一化为概率，保持输入形状。"""
    axis = normalize_axis(axis, len(data.shape))
    if data.shape[axis] == 0: raise ValueError("Softmax 的计算轴不能没有元素")
    backend = get_backend(data.storage.device)
    values = backend.softmax(data.storage, data.shape, axis)
    return values, data.shape, partial(_softmax_backward, probabilities=values, shape=data.shape, axis=axis)


def _softmax_backward(gradient: Storage, probabilities: Storage, shape: Shape, axis: int) -> tuple[Storage, ...]:
    """输入梯度为 p * (g - sum(g * p))，求和沿同一个轴进行。"""
    backend = get_backend(gradient.device)
    weighted = backend.multiply(gradient, probabilities)
    totals = backend.sum(weighted, shape, axis)
    totals = backend.broadcast(totals, reduced_shape(shape, axis, keepdims=True), shape)
    centered = backend.subtract(gradient, totals)
    return (backend.multiply(probabilities, centered),)


def batch_norm2d(data: Operand, weight: Operand, bias: Operand, running_mean: Storage, running_var: Storage, *, train: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Computation:
    """NCHW 批归一化；训练更新运行统计，预测使用固定统计，两种模式都可求导。"""
    if len(data.shape) != 4 or any(size <= 0 for size in data.shape[1:]): raise ValueError("批归一化输入必须为 NCHW 四维数据，通道、高和宽须大于零")
    batch, channels, height, width = data.shape
    if weight.shape != (channels,) or bias.shape != (channels,): raise ValueError("缩放和平移参数的形状必须等于通道数")
    if running_mean.size != channels or running_var.size != channels: raise ValueError("运行统计量的长度必须等于通道数")
    if eps <= 0: raise ValueError("eps 必须大于零")
    count = batch * height * width
    if train and count <= 1: raise ValueError("训练时每个通道必须包含至少两个元素")
    backend = get_backend(data.storage, weight.storage, bias.storage, running_mean, running_var)
    averages, variances = backend.channel_stats(data.storage, data.shape) if train else (running_mean, running_var)
    values, normalized, inverse = backend.batch_norm2d(data.storage, weight.storage, bias.storage, data.shape, averages, variances, eps)
    if train:
        backend.copy_into(running_mean, backend.add(backend.scale(running_mean, 1 - momentum), backend.scale(averages, momentum)))
        backend.copy_into(running_var, backend.add(backend.scale(running_var, 1 - momentum), backend.scale(variances, momentum * count / (count - 1))))
    rule = partial(_batch_norm2d_backward, normalized=normalized, inverse=inverse, weight=weight.storage.copy(), shape=data.shape, train=train)
    return values, data.shape, rule


def _batch_norm2d_backward(gradient: Storage, normalized: Storage, inverse: Storage, weight: Storage, shape: Shape, train: bool) -> tuple[Storage, ...]:
    """使用前向快照，后续运行统计更新不会改变旧计算图的梯度。"""
    backend = get_backend(gradient.device)
    return backend.batch_norm2d_backward(gradient, normalized, inverse, weight, shape, train)


def layer_norm(data: Operand, weight: Operand, bias: Operand, *, eps: float = 1e-5) -> Computation:
    """沿最后一轴归一化；每组单独计算均值和总体方差，训练与预测行为相同。"""
    if not data.shape or data.shape[-1] <= 0: raise ValueError("层归一化的最后一轴必须包含至少一个特征")
    width = data.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,): raise ValueError("缩放和平移参数的形状必须等于最后一轴的特征数")
    if eps <= 0: raise ValueError("eps 必须大于零")
    backend = get_backend(data.storage, weight.storage, bias.storage)
    values, normalized, inverse = backend.layer_norm(data.storage, weight.storage, bias.storage, width, eps)
    rule = partial(_layer_norm_backward, normalized=normalized, inverse=inverse, weight=weight.storage.copy())
    return values, data.shape, rule


def _layer_norm_backward(gradient: Storage, normalized: Storage, inverse: Storage, weight: Storage) -> tuple[Storage, ...]:
    """反向使用前向快照，后续输入或权重更新不会改变旧图的梯度。"""
    backend = get_backend(gradient.device)
    return backend.layer_norm_backward(gradient, normalized, inverse, weight)
