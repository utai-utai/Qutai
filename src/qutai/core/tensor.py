from collections.abc import Iterable
from functools import partial
from math import prod
from typing import overload

from ..backends.device import get_backend, transfer
from ..backends.storage import Storage
from . import grad as autograd
from . import operations
from .shape import flatten, infer_shape, unflatten
from .types import Backward, Computation, Data, Index, Indices, IntegerData, Result, Shape


class Tensor:
    def __init__(self, data: Data | Storage, *, shape: Shape | None = None, requires_grad: bool = False, device: str | None = None, dtype: str | None = None) -> None:
        device = device if device is not None else data.device if isinstance(data, Storage) else "cpu"
        dtype = dtype if dtype is not None else data.dtype if isinstance(data, Storage) else "float64"
        backend = get_backend(device)
        if dtype not in ("float32", "float64"): raise ValueError("实数 Tensor 只支持 float32 或 float64")
        if isinstance(data, Storage):
            if data.device != device or data.dtype != dtype: raise ValueError("输入存储的设备或精度与 Tensor 不一致")
            self.shape: Shape = (data.size,) if shape is None else shape
            self.storage: Storage = data
        else:
            self.shape = infer_shape(data) if shape is None else shape
            self.storage = backend.storage(flatten(data), dtype=dtype)
        if any(size < 0 for size in self.shape) or prod(self.shape) != self.storage.size:
            raise ValueError("元素数量与 Tensor 形状不匹配")
        self.requires_grad: bool = requires_grad
        self.grad: Tensor | None = None
        self.parents: tuple[Tensor, ...] = ()
        self.backward_rule: Backward | None = None

    @property
    def device(self) -> str:
        return self.storage.device

    @property
    def dtype(self) -> str:
        return self.storage.dtype

    @property
    def values(self) -> Storage:
        """CPU 调试用的可变数值存储；计算统一使用 storage。"""
        if self.device != "cpu": raise ValueError("逐元素访问仅适用于 CPU，请使用 tolist 或 item 读取结果")
        return self.storage

    @values.setter
    def values(self, data: Iterable[float]) -> None:
        self.storage = get_backend(self.device).storage(data, dtype=self.dtype)

    @property
    def size(self) -> int:
        """元素总数。"""
        return self.storage.size

    @property
    def ndim(self) -> int:
        """张量的维数。"""
        return len(self.shape)

    @property
    def T(self) -> Tensor:
        """反转全部轴；批量矩阵只交换末两轴时使用 transpose。"""
        return self.transpose()

    def __len__(self) -> int:
        if not self.shape: raise TypeError("标量张量没有长度")
        return self.shape[0]

    def __getitem__(self, index: Index) -> Tensor:
        return _result(operations.select(self, index), (self,))

    def __add__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.add(self, other), (self, other))

    def __radd__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.add(other, self), (other, self))

    def __sub__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.subtract(self, other), (self, other))

    def __rsub__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.subtract(other, self), (other, self))

    def __mul__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.multiply(self, other), (self, other))

    def __rmul__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.multiply(other, self), (other, self))

    def __truediv__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.divide(self, other), (self, other))

    def __rtruediv__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.divide(other, self), (other, self))

    def __matmul__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.matmul(self, other), (self, other))

    def __rmatmul__(self, other: Data | Tensor) -> Tensor:
        other = _as_tensor(other, self)
        return _result(operations.matmul(other, self), (other, self))

    def __neg__(self) -> Tensor:
        return _result(operations.negate(self), (self,))

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return _result(operations.sum(self, axis=axis, keepdims=keepdims), (self,))

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return _result(operations.mean(self, axis=axis, keepdims=keepdims), (self,))

    def reshape(self, shape: Shape) -> Tensor:
        return _result(operations.reshape(self, shape), (self,))

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        return _result(operations.transpose(self, axes), (self,))

    def square(self) -> Tensor:
        return _result(operations.square(self), (self,))

    def exp(self) -> Tensor:
        return _result(operations.exp(self), (self,))

    def log(self) -> Tensor:
        return _result(operations.log(self), (self,))

    def argmax(self, axis: int | None = None, keepdims: bool = False) -> Indices:
        return operations.argmax(self, axis=axis, keepdims=keepdims)

    def tolist(self) -> Result:
        """返回普通数字或嵌套列表。"""
        return unflatten(self.storage.tolist(), self.shape)

    def item(self) -> float:
        """取出唯一的元素。"""
        if self.size != 1: raise ValueError("只有包含一个元素的值才能取出单个数字")
        return self.storage.tolist()[0]

    def to(self, device: str | None = None, *, dtype: str | None = None, copy: bool = False) -> Tensor:
        """返回转换后的 Tensor 并保留求导关系；相同配置默认返回自身。"""
        if dtype is not None and dtype not in ("float32", "float64"): raise ValueError("实数 Tensor 只支持 float32 或 float64")
        values = transfer(self.storage, device, dtype, copy=copy)
        if values is self.storage: return self
        rule = partial(_to_backward, device=self.device, dtype=self.dtype)
        return autograd.record(Tensor(values, shape=self.shape), (self,), rule)

    def zero_grad(self) -> None:
        """清除此前保存的梯度。"""
        self.grad = None

    def backward(self, gradient: Data | Tensor | None = None) -> None:
        """从当前结果反向传播；非单元素结果需要同形状的输出梯度。"""
        if not self.requires_grad:
            raise ValueError("当前 Tensor 未开启梯度记录")
        if gradient is None:
            if self.size != 1:
                raise ValueError("非单元素结果需要提供同形状的输出梯度")
            seed = get_backend(self.device).full(self.size, 1.0, dtype=self.dtype)
        else:
            gradient = _as_tensor(gradient, self)
            if gradient.shape != self.shape:
                raise ValueError("输出梯度的形状必须与当前 Tensor 相同")
            if gradient.device != self.device or gradient.dtype != self.dtype: raise ValueError("输出梯度的设备和精度必须与当前 Tensor 相同")
            seed = gradient.storage

        for tensor, values in autograd.backward(self, seed).items():
            if tensor.grad is not None:
                values = get_backend(tensor.device).add(tensor.grad.storage, values)
            tensor.grad = Tensor(values.copy(), shape=tensor.shape, device=tensor.device, dtype=tensor.dtype)

    def __repr__(self) -> str:
        return f"Tensor({self.tolist()!r})"


def _to_backward(gradient: Storage, *, device: str, dtype: str) -> tuple[Storage, ...]:
    return (transfer(gradient, device, dtype),)


def _as_tensor(data: Data | Tensor, like: Tensor | None = None) -> Tensor:
    """普通输入沿用参考 Tensor 的设备和精度，已有对象不复制。"""
    if isinstance(data, Tensor):
        if like is not None: get_backend(like.storage, data.storage)
        return data
    return Tensor(data, device=None if like is None else like.device, dtype=None if like is None else like.dtype)


def _pair(left: Data | Tensor, right: Data | Tensor) -> tuple[Tensor, Tensor]:
    like = left if isinstance(left, Tensor) else right if isinstance(right, Tensor) else None
    return _as_tensor(left, like), _as_tensor(right, like)


def _result(computation: Computation, parents: tuple[Tensor, ...]) -> Tensor:
    """将运算结果包装为 Tensor，并记录输入与局部反向规则。"""
    values, shape, rule = computation
    return autograd.record(Tensor(values, shape=shape, device=values.device, dtype=values.dtype), parents, rule)


def _output(result: Tensor, *inputs: Data | Tensor | None) -> Result | Tensor:
    """有 Tensor 输入就保留 Tensor，否则返回普通数字或列表。"""
    return result if any(isinstance(data, Tensor) for data in inputs) else result.tolist()


def add(left: Data | Tensor, right: Data | Tensor) -> Result | Tensor:
    """广播相加，保留求导关系。"""
    a, b = _pair(left, right)
    return _output(a + b, left, right)


def subtract(left: Data | Tensor, right: Data | Tensor) -> Result | Tensor:
    """广播相减，保留求导关系。"""
    a, b = _pair(left, right)
    return _output(a - b, left, right)


def multiply(left: Data | Tensor, right: Data | Tensor) -> Result | Tensor:
    """广播相乘，保留求导关系。"""
    a, b = _pair(left, right)
    return _output(a * b, left, right)


def divide(left: Data | Tensor, right: Data | Tensor) -> Result | Tensor:
    """广播相除，保留求导关系。"""
    a, b = _pair(left, right)
    return _output(a / b, left, right)


def matmul(left: Data | Tensor, right: Data | Tensor) -> Result | Tensor:
    """计算向量、矩阵或批量矩阵乘积，仅在批次维度广播。"""
    a, b = _pair(left, right)
    return _output(a @ b, left, right)


def negate(data: Data | Tensor) -> Result | Tensor:
    """逐元素取负。"""
    return _output(-_as_tensor(data), data)


def square(data: Data | Tensor) -> Result | Tensor:
    """逐元素平方。"""
    return _output(_as_tensor(data).square(), data)


def relu(data: Data | Tensor) -> Result | Tensor:
    """逐元素保留正数，将其余值置零。"""
    tensor = _as_tensor(data)
    return _output(_result(operations.relu(tensor), (tensor,)), data)


def gelu(data: Data | Tensor) -> Result | Tensor:
    """逐元素计算 GELU：x 乘以标准正态分布在 x 处的累积概率。"""
    tensor = _as_tensor(data)
    return _output(_result(operations.gelu(tensor), (tensor,)), data)


def exp(data: Data | Tensor) -> Result | Tensor:
    """逐元素计算以 e 为底的指数。"""
    return _output(_as_tensor(data).exp(), data)


def log(data: Data | Tensor) -> Result | Tensor:
    """逐元素计算自然对数，输入必须大于零。"""
    return _output(_as_tensor(data).log(), data)


def sum(data: Data | Tensor, axis: int | None = None, keepdims: bool = False) -> Result | Tensor:
    """计算全部元素或指定轴上的和，可保留归约维度。"""
    return _output(_as_tensor(data).sum(axis=axis, keepdims=keepdims), data)


def mean(data: Data | Tensor, axis: int | None = None, keepdims: bool = False) -> Result | Tensor:
    """计算全部元素或指定轴上的平均值。"""
    return _output(_as_tensor(data).mean(axis=axis, keepdims=keepdims), data)


def argmax(data: Data | Tensor, axis: int | None = None, keepdims: bool = False) -> Indices:
    """返回最大值的第一个整数编号；始终返回普通数据，不参与求导。"""
    return _as_tensor(data).argmax(axis=axis, keepdims=keepdims)


def reshape(data: Data | Tensor, shape: Shape) -> Result | Tensor:
    """保持元素顺序重新组织数据，支持一个 -1。"""
    return _output(_as_tensor(data).reshape(shape), data)


@overload
def stack(data: Iterable[Tensor], axis: int = 0) -> Tensor: ...


@overload
def stack(data: Iterable[Data | Tensor], axis: int = 0) -> Result | Tensor: ...


def stack(data: Iterable[Data | Tensor], axis: int = 0) -> Result | Tensor:
    """沿新增的轴组合同形状数据；有 Tensor 输入时保留计算关系。"""
    inputs = tuple(data)
    tensors = tuple(_as_tensor(item) for item in inputs)
    return _output(_result(operations.stack(tensors, axis=axis), tensors), *inputs)


def transpose(data: Data | Tensor, axes: tuple[int, ...] | None = None) -> Result | Tensor:
    """按照指定轴顺序转置，默认反转全部轴。"""
    return _output(_as_tensor(data).transpose(axes), data)


def select(data: Data | Tensor, index: Index) -> Result | Tensor:
    """按索引读取数据；反向时将梯度加回原来的位置。"""
    return _output(_as_tensor(data)[index], data)


def embedding(indices: IntegerData, weight: Data | Tensor) -> Result | Tensor:
    """用整数编号查权重表，保留编号的形状并在末尾增加特征轴。"""
    tensor = _as_tensor(weight)
    return _output(_result(operations.embedding(indices, tensor), (tensor,)), weight)


def softmax(data: Data | Tensor, axis: int = -1) -> Result | Tensor:
    """沿指定轴将得分归一化为概率，保持输入形状。"""
    tensor = _as_tensor(data)
    return _output(_result(operations.softmax(tensor, axis=axis), (tensor,)), data)


def conv2d(data: Data | Tensor, weight: Data | Tensor, bias: Data | Tensor | None = None, *, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0) -> Result | Tensor:
    """NCHW 输入与 OIHW 卷积核做卷积，可指定偏置、步长和零填充。"""
    like = next((item for item in (data, weight, bias) if isinstance(item, Tensor)), None)
    tensor, kernel = _as_tensor(data, like), _as_tensor(weight, like)
    offset = None if bias is None else _as_tensor(bias, like)
    parents = (tensor, kernel) if offset is None else (tensor, kernel, offset)
    computation = operations.conv2d(tensor, kernel, offset, stride=stride, padding=padding)
    return _output(_result(computation, parents), data, weight, bias)


def max_pool2d(data: Data | Tensor, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] | None = None, padding: int | tuple[int, int] = 0) -> Result | Tensor:
    """NCHW 数据按窗口取最大值，默认步长等于窗口大小。"""
    tensor = _as_tensor(data)
    computation = operations.max_pool2d(tensor, kernel_size, stride=stride, padding=padding)
    return _output(_result(computation, (tensor,)), data)


def batch_norm2d(data: Data | Tensor, weight: Data | Tensor, bias: Data | Tensor, running_mean: Storage | list[float], running_var: Storage | list[float], *, train: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Result | Tensor:
    """NCHW 批归一化；训练更新运行统计，预测使用固定统计，两种模式都可求导。"""
    like = next((item for item in (data, weight, bias) if isinstance(item, Tensor)), None)
    tensor, scale, shift = _as_tensor(data, like), _as_tensor(weight, like), _as_tensor(bias, like)
    backend = get_backend(tensor.device)
    average = running_mean if isinstance(running_mean, Storage) else backend.storage(running_mean, dtype=tensor.dtype)
    variance = running_var if isinstance(running_var, Storage) else backend.storage(running_var, dtype=tensor.dtype)
    computation = operations.batch_norm2d(tensor, scale, shift, average, variance, train=train, momentum=momentum, eps=eps)
    if not isinstance(running_mean, Storage): running_mean[:] = average.tolist()
    if not isinstance(running_var, Storage): running_var[:] = variance.tolist()
    return _output(_result(computation, (tensor, scale, shift)), data, weight, bias)


def layer_norm(data: Data | Tensor, weight: Data | Tensor, bias: Data | Tensor, *, eps: float = 1e-5) -> Result | Tensor:
    """沿最后一轴归一化；每组单独计算均值和总体方差，训练与预测行为相同。"""
    like = next((item for item in (data, weight, bias) if isinstance(item, Tensor)), None)
    tensor, scale, shift = _as_tensor(data, like), _as_tensor(weight, like), _as_tensor(bias, like)
    computation = operations.layer_norm(tensor, scale, shift, eps=eps)
    return _output(_result(computation, (tensor, scale, shift)), data, weight, bias)
