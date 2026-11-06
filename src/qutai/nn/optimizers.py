import json
from collections.abc import Iterable
from math import isfinite
from pathlib import Path
from typing import Self

from ..backends.device import get_backend, transfer
from ..backends.storage import Storage
from ..core.tensor import Tensor
from ..core.migration import ParameterSource


def _parameters(source: ParameterSource | Iterable[Tensor]) -> tuple[Tensor, ...]:
    """接受模型、单层或参数集合，共享参数只保留一次。"""
    if isinstance(source, ParameterSource): source = source.parameters()
    return tuple(dict.fromkeys(source))


def clip_grad_norm(parameters: ParameterSource | Iterable[Tensor], max_norm: float = 1.0) -> float:
    """把全部已有梯度的 L2 范数限制在 max_norm，返回裁剪前的范数。"""
    if not isfinite(max_norm) or max_norm < 0: raise ValueError("梯度范数上限必须是非负有限数")
    gradients = [parameter.grad for parameter in _parameters(parameters) if parameter.grad is not None]
    backend = get_backend(gradients[0].device if gradients else "cpu")
    if any(gradient.device != gradients[0].device for gradient in gradients): raise ValueError("参与裁剪的梯度必须位于同一设备")
    norm = backend.global_norm(tuple(gradient.storage for gradient in gradients))
    if not isfinite(norm): raise ValueError("梯度包含非有限值，无法裁剪")
    if norm > max_norm:
        scale = max_norm / norm
        for gradient in gradients:
            backend.copy_into(gradient.storage, backend.scale(gradient.storage, scale))
    return norm


class Optimizer:
    """共用参数收集、清梯度与状态文件；具体更新规则由子类实现。"""
    _settings: tuple[str, ...] = ()
    _buffers: tuple[str, ...] = ()

    def __init__(self, parameters: ParameterSource | Iterable[Tensor]) -> None:
        if isinstance(parameters, ParameterSource): parameters = parameters.parameters()
        items = tuple(parameters)
        self.param_groups = [{**group, "params": _parameters(group["params"])} for group in items] if items and isinstance(items[0], dict) else [{"params": _parameters(items)}]
        self.parameters = tuple(p for group in self.param_groups for p in group["params"])
        if len(set(self.parameters)) != len(self.parameters): raise ValueError("同一参数不能出现在多个优化器参数组中")

    def to(self) -> Self:
        """动量和移动平均跟随各自参数的设备与精度，参数对象和计步不变。"""
        updates = [(state, parameter, transfer(value, parameter.device, parameter.dtype))
                   for name in self._buffers for state in (getattr(self, name),) for parameter, value in state.items()]
        for state, parameter, value in updates: state[parameter] = value
        return self

    def zero_grad(self) -> None:
        """清除所有参数此前保存的梯度。"""
        for parameter in self.parameters:
            parameter.zero_grad()

    def save(self, path: str | Path) -> None:
        """保存超参数和每个参数的更新状态；模型数值由 model.save 单独保存。"""
        data = {"type": type(self).__name__, "settings": {name: getattr(self, name) for name in self._settings},
                "shapes": [parameter.shape for parameter in self.parameters], "state": self._state(),
                "groups": [{**{key: value for key, value in group.items() if key != "params"}, "params": [self.parameters.index(p) for p in group["params"]]} for group in self.param_groups]}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> Self:
        """按当前参数顺序恢复状态，保留参数对象和梯度；返回优化器自身。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["type"] != type(self).__name__: raise ValueError("文件的优化器类型与当前优化器不一致")
        if data["shapes"] != [list(parameter.shape) for parameter in self.parameters]:
            raise ValueError("文件的参数数量或形状与当前优化器不一致")
        if len(data["state"]) != len(self.parameters): raise ValueError("优化器状态数量与参数数量不一致")
        # 先在临时对象完成校验，失败时不改变当前优化器。
        groups = [{**group, "params": [self.parameters[i] for i in group["params"]]} for group in data.get("groups", [])]
        restored = type(self)(groups or self.parameters, **data["settings"])
        restored._restore(data["state"])
        self.__dict__.update(restored.__dict__)
        return self

    def _state(self) -> list:
        """按参数顺序收集更新状态；尚未参与更新的参数用 None 表示。"""
        raise NotImplementedError

    def _restore(self, state: list) -> None:
        """校验并绑定到当前参数；文件中不使用 Python 对象编号。"""
        raise NotImplementedError


class SGD(Optimizer):
    _buffers = ("velocities",)
    _settings = ("learning_rate", "momentum", "weight_decay")

    def __init__(self, parameters: ParameterSource | Iterable[Tensor], *, learning_rate: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        """接收模型、单层或参数集合；可选动量和权重衰减。"""
        super().__init__(parameters)
        if learning_rate < 0 or momentum < 0 or weight_decay < 0: raise ValueError("学习率、动量和权重衰减不能为负数")
        self.learning_rate = learning_rate
        self.momentum, self.weight_decay = momentum, weight_decay
        self.velocities: dict[Tensor, Storage] = {}

    def step(self) -> None:
        """可选权重衰减和动量，再按梯度下降更新参数。"""
        self.to()
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None: continue
            backend = get_backend(parameter.storage, gradient.storage)
            parameter.storage, velocity = backend.sgd_step(parameter.storage, gradient.storage, self.velocities.get(parameter),
                                                          learning_rate=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
            if velocity is not None: self.velocities[parameter] = velocity

    def _state(self) -> list:
        return [self.velocities[parameter].tolist() if parameter in self.velocities else None for parameter in self.parameters]

    def _restore(self, state: list) -> None:
        for parameter, velocity in zip(self.parameters, state, strict=True):
            if velocity is None: continue
            if len(velocity) != parameter.size: raise ValueError("动量状态长度与参数元素数量不一致")
            self.velocities[parameter] = get_backend(parameter.device).storage(velocity, dtype=parameter.dtype)


class AdamW(Optimizer):
    """用梯度的一阶、二阶移动平均自适应更新，并独立执行权重衰减。"""
    _settings = ("learning_rate", "betas", "eps", "weight_decay")
    _buffers = ("averages", "square_averages")

    def __init__(self, parameters: ParameterSource | Iterable[Tensor], *, learning_rate: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        super().__init__(parameters)
        if learning_rate < 0 or weight_decay < 0 or eps < 0: raise ValueError("学习率、权重衰减和 eps 不能为负数")
        if len(betas) != 2 or any(beta < 0 or beta >= 1 for beta in betas): raise ValueError("betas 必须包含两个不小于 0 且小于 1 的数")
        self.learning_rate, self.betas = learning_rate, tuple(betas)
        self.eps, self.weight_decay = eps, weight_decay
        self.steps: dict[Tensor, int] = {}
        self.averages: dict[Tensor, Storage] = {}
        self.square_averages: dict[Tensor, Storage] = {}

    def step(self) -> None:
        """按组更新；CUDA 可原地更新，参数与状态身份保持。"""
        self.to()
        for group in self.param_groups:
            settings = {name: group.get(name, getattr(self, name)) for name in self._settings}
            if settings["learning_rate"] < 0 or settings["eps"] < 0 or settings["weight_decay"] < 0: raise ValueError("优化器参数不能为负数")
            if len(settings["betas"]) != 2 or any(not 0 <= b < 1 for b in settings["betas"]): raise ValueError("betas 必须在 [0, 1) 内")
            for parameter in group["params"]:
                if parameter.grad is None: continue
                backend = get_backend(parameter.storage, parameter.grad.storage)
                step = self.steps.get(parameter, 0) + 1
                previous, squares = self.averages.get(parameter), self.square_averages.get(parameter)
                if previous is None: previous = backend.full(parameter.size, dtype=parameter.dtype)
                if squares is None: squares = backend.full(parameter.size, dtype=parameter.dtype)
                update = getattr(backend, "adamw_step_into", backend.adamw_step)
                values, average, square_average = update(parameter.storage, parameter.grad.storage, previous, squares, step=step, **settings)
                self.steps[parameter], self.averages[parameter], self.square_averages[parameter] = step, average, square_average
                parameter.storage = values

    def _state(self) -> list:
        return [None if parameter not in self.steps else
                {"step": self.steps[parameter], "average": self.averages[parameter].tolist(), "square_average": self.square_averages[parameter].tolist()}
                for parameter in self.parameters]

    def _restore(self, state: list) -> None:
        for parameter, item in zip(self.parameters, state, strict=True):
            if item is None: continue
            if type(item["step"]) is not int or item["step"] < 1: raise ValueError("AdamW 的更新次数必须是正整数")
            if len(item["average"]) != parameter.size or len(item["square_average"]) != parameter.size:
                raise ValueError("AdamW 状态长度与参数元素数量不一致")
            if any(value < 0 for value in item["square_average"]): raise ValueError("梯度平方的移动平均不能为负数")
            self.steps[parameter] = item["step"]
            backend = get_backend(parameter.device)
            self.averages[parameter], self.square_averages[parameter] = (backend.storage(item[name], dtype=parameter.dtype) for name in ("average", "square_average"))
