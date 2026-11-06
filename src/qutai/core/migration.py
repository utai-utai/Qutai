from typing import Protocol, runtime_checkable

from ..backends.device import get_backend, transfer
from ..backends.storage import Storage
from .tensor import Tensor


@runtime_checkable
class ParameterSource(Protocol):
    """能提供参数的模型或层，自定义对象无需强制继承。"""
    def parameters(self) -> tuple[Tensor, ...]: ...


def move[T: ParameterSource](model: T, device: str | None = None, dtype: str | None = None) -> T:
    """原地迁移参数、已有梯度和显式 buffers；共享对象只转换一次。"""
    if dtype is not None and dtype not in ("float32", "float64"): raise ValueError("模型参数只支持 float32 或 float64")
    if device is not None: get_backend(device)
    placements = []
    for parameter in dict.fromkeys(model.parameters()):
        if parameter.parents: raise ValueError("模型迁移需要叶子参数，请在前向计算之前迁移")
        placements.append((parameter, "storage", parameter.storage))
        if parameter.grad is not None: placements.append((parameter.grad, "storage", parameter.grad.storage))
    pending, seen = [model], set()
    while pending:
        layer = pending.pop()
        if id(layer) in seen: continue
        seen.add(id(layer))
        if hasattr(layer, "buffers"):
            placements.extend((layer, name, value) for name, value in layer.buffers().items())
        for value in getattr(layer, "__dict__", {}).values():
            pending.extend(child for child in (value if isinstance(value, (tuple, list)) else (value,)) if hasattr(child, "parameters"))
    converted: dict[int, Storage] = {}
    # 全部转换成功后再修改对象，失败时保留模型原有状态。
    for _, _, value in placements:
        if id(value) not in converted: converted[id(value)] = transfer(value, device, dtype)
    for owner, name, value in placements:
        setattr(owner, name, converted[id(value)])
    return model
