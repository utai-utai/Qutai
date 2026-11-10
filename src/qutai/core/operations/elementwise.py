from collections.abc import Callable
from functools import partial

from ...backends.device import get_backend
from ...backends.storage import Storage
from ..shape import broadcast_shape
from ..types import Computation, Operand


def _binary(left: Operand, right: Operand, operation: str, backward: Callable[[Storage, Operand, Operand], tuple[Storage, ...]]) -> Computation:
    """按输入设备广播，调用同名后端运算并绑定反向规则。"""
    backend = get_backend(left.storage, right.storage)
    shape = broadcast_shape(left.shape, right.shape)
    a = backend.broadcast(left.storage, left.shape, shape)
    b = backend.broadcast(right.storage, right.shape, shape)
    return getattr(backend, operation)(a, b), shape, partial(backward, left=left, right=right)


def add(left: Operand, right: Operand) -> Computation:
    return _binary(left, right, "add", _add_backward)


def _add_backward(gradient: Storage, left: Operand, right: Operand) -> tuple[Storage, ...]:
    """两边都接收 g，再把广播位置的梯度求和。"""
    backend = get_backend(gradient.device)
    shape = broadcast_shape(left.shape, right.shape)
    return backend.unbroadcast(gradient, shape, left.shape), backend.unbroadcast(gradient, shape, right.shape)


def subtract(left: Operand, right: Operand) -> Computation:
    return _binary(left, right, "subtract", _subtract_backward)


def _subtract_backward(gradient: Storage, left: Operand, right: Operand) -> tuple[Storage, ...]:
    """左边接收 g，右边接收 -g。"""
    a, b = _add_backward(gradient, left, right)
    return a, get_backend(gradient.device).negate(b)


def multiply(left: Operand, right: Operand) -> Computation:
    return _binary(left, right, "multiply", _multiply_backward)


def _multiply_backward(gradient: Storage, left: Operand, right: Operand) -> tuple[Storage, ...]:
    """两边分别接收 g*b、g*a，并还原广播。"""
    backend = get_backend(gradient.device)
    shape = broadcast_shape(left.shape, right.shape)
    a = backend.broadcast(left.storage, left.shape, shape)
    b = backend.broadcast(right.storage, right.shape, shape)
    return (backend.unbroadcast(backend.multiply(gradient, b), shape, left.shape),
            backend.unbroadcast(backend.multiply(gradient, a), shape, right.shape))


def divide(left: Operand, right: Operand) -> Computation:
    return _binary(left, right, "divide", _divide_backward)


def divide_scalar(data: Operand, value: float) -> Computation:
    backend = get_backend(data.storage)
    return backend.divide_scalar(data.storage, value), data.shape, partial(_divide_scalar_backward, value=value)


def _divide_scalar_backward(gradient: Storage, value: float) -> tuple[Storage, ...]:
    return (get_backend(gradient).divide_scalar(gradient, value),)


def _divide_backward(gradient: Storage, left: Operand, right: Operand) -> tuple[Storage, ...]:
    """两边分别接收 g/b、-g*a/b²，并还原广播。"""
    backend = get_backend(gradient.device)
    shape = broadcast_shape(left.shape, right.shape)
    a = backend.broadcast(left.storage, left.shape, shape)
    b = backend.broadcast(right.storage, right.shape, shape)
    ga = backend.divide(gradient, b)
    gb = backend.negate(backend.multiply(ga, backend.divide(a, b)))
    return backend.unbroadcast(ga, shape, left.shape), backend.unbroadcast(gb, shape, right.shape)


def _unary(data: Operand, operation: str, backward: Callable[[Storage, Operand], tuple[Storage, ...]]) -> Computation:
    """调用命名的一元后端运算，保留形状。"""
    values = getattr(get_backend(data.storage.device), operation)(data.storage)
    return values, data.shape, partial(backward, tensor=data)


def negate(data: Operand) -> Computation:
    return _unary(data, "negate", _negate_backward)


def _negate_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    return (get_backend(gradient.device).negate(gradient),)


def square(data: Operand) -> Computation:
    return _unary(data, "square", _square_backward)


def _square_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    backend = get_backend(gradient.device)
    return (backend.scale(backend.multiply(gradient, tensor.storage), 2.0),)


def relu(data: Operand) -> Computation:
    return _unary(data, "relu", _relu_backward)


def _relu_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    return (get_backend(gradient.device).relu_backward(gradient, tensor.storage),)


def gelu(data: Operand) -> Computation:
    return _unary(data, "gelu", _gelu_backward)


def _gelu_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    return (get_backend(gradient.device).gelu_backward(gradient, tensor.storage),)


def exp(data: Operand) -> Computation:
    return _unary(data, "exp", _exp_backward)


def _exp_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    backend = get_backend(gradient.device)
    return (backend.multiply(gradient, backend.exp(tensor.storage)),)


def log(data: Operand) -> Computation:
    return _unary(data, "log", _log_backward)


def tanh(data): return _unary(data, 'tanh', _tanh_backward)
def atan(data): return _unary(data, 'atan', _atan_backward)
def _tanh_backward(gradient, tensor): return (get_backend(gradient.device).tanh_backward(gradient, tensor.storage),)
def _atan_backward(gradient, tensor): return (get_backend(gradient.device).atan_backward(gradient, tensor.storage),)


def _log_backward(gradient: Storage, tensor: Operand) -> tuple[Storage, ...]:
    return (get_backend(gradient.device).divide(gradient, tensor.storage),)
