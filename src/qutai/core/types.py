from collections.abc import Callable
from random import Random
from types import EllipsisType
from typing import Protocol, Self

from ..backends.storage import Storage


type Shape = tuple[int, ...]
type Data = int | float | list[Data] | tuple[Data, ...]
type IntegerData = int | list[IntegerData] | tuple[IntegerData, ...]
type Nested[T] = T | list[Nested[T]]
type Batch[T] = tuple[T, list[int]] | tuple[list[list[int]], list[list[int]]] | tuple[T, list[float], T]
type Result = Nested[float]
type Indices = Nested[int]
type Backward = Callable[[Storage], tuple[Storage, ...]]
type Computation = tuple[Storage, Shape, Backward]
type Selector = int | slice | EllipsisType | None
type Index = Selector | tuple[Selector, ...]


class GenerationMethod[T](Protocol):
    """生成数据流需要的批次准备接口；不依赖具体生成方法或 Tensor 类。"""
    def prepare(self, images: T, random: Random) -> tuple[T, list[float], T]: ...


class LayerLike[T](Protocol):
    """模型容器需要的层接口，经典层和量子层都可直接满足。"""
    def __call__(self, data: Data | T) -> T: ...

    def parameters(self) -> tuple[T, ...]: ...


class Operand(Protocol):
    """运算读取的数据属性，不依赖具体的 Tensor 类。"""
    shape: Shape
    storage: Storage
    requires_grad: bool

    @property
    def size(self) -> int: ...

    @property
    def ndim(self) -> int: ...


class GradNode(Protocol):
    """反向引擎需要的节点字段；不依赖具体的 Tensor 类。"""
    requires_grad: bool
    parents: tuple[Self, ...]
    backward_rule: Backward | None
