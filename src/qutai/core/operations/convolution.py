from functools import partial

from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import pair, window_shape
from ..types import Computation, Operand


def conv2d(data: Operand, weight: Operand, bias: Operand | None = None, *, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0) -> Computation:
    """NCHW 输入与 OIHW 卷积核做卷积，可指定偏置、步长和零填充。"""
    stride, padding = pair(stride), pair(padding)
    if len(weight.shape) != 4: raise ValueError("卷积核必须为（输出通道，输入通道，高，宽）四维数据")
    spatial = window_shape(data.shape, (weight.shape[2], weight.shape[3]), stride, padding)
    if weight.shape[0] <= 0 or weight.shape[1] != data.shape[1]: raise ValueError("卷积核的通道数与输入不匹配")
    if bias is not None and bias.shape != (weight.shape[0],): raise ValueError("偏置必须为每个输出通道提供一个数值")
    shape = (spatial[0], weight.shape[0], *spatial[2:])
    backend = get_backend(data.storage, weight.storage, *(() if bias is None else (bias.storage,)))
    values = backend.conv2d(data.storage, weight.storage, None if bias is None else bias.storage, data.shape, weight.shape, stride, padding)
    rule = partial(_conv2d_backward, tensor=data, kernel=weight, stride=stride, padding=padding, has_bias=bias is not None)
    return values, shape, rule


def _conv2d_backward(gradient: Storage, tensor: Operand, kernel: Operand, stride: tuple[int, int], padding: tuple[int, int], has_bias: bool) -> tuple[Storage, ...]:
    """输入接收 g*w，卷积核接收 g*x，偏置接收各批次与空间位置的 g 之和。"""
    backend = get_backend(gradient.device)
    gradients = backend.conv2d_backward(gradient, tensor.storage, kernel.storage, tensor.shape, kernel.shape, stride, padding, input_grad=tensor.requires_grad)
    return gradients if has_bias else gradients[:2]


def max_pool2d(data: Operand, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] | None = None, padding: int | tuple[int, int] = 0) -> Computation:
    """NCHW 数据按窗口取最大值，默认步长等于窗口大小。"""
    kernel_size, padding = pair(kernel_size), pair(padding)
    stride = kernel_size if stride is None else pair(stride)
    shape = window_shape(data.shape, kernel_size, stride, padding)
    if any(pad > size // 2 for pad, size in zip(padding, kernel_size, strict=True)): raise ValueError("池化填充不能超过窗口大小的一半")
    backend = get_backend(data.storage.device)
    values, indices = backend.max_pool2d(data.storage, data.shape, kernel_size, stride, padding)
    return values, shape, partial(_max_pool2d_backward, indices=indices, size=data.size)


def _max_pool2d_backward(gradient: Storage, indices: Storage, size: int) -> tuple[Storage, ...]:
    """仅向各窗口最大值的位置传递梯度。"""
    backend = get_backend(gradient.device)
    return (backend.max_pool2d_backward(gradient, indices, size),)
