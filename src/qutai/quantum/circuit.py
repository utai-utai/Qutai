from collections.abc import Iterable
from functools import partial
from typing import Self, overload

from ..backends.device import get_backend
from ..core.grad import record
from ..core.tensor import Tensor
from .gates import Gate
from .grad import adjoint_backward, adjoint_probabilities_backward, parameter_shift_backward, parameter_shift_probabilities_backward
from .observables import Operator, as_observable, expectation_values
from .state import State, selected_wires


class Circuit:
    """保存量子门；run 返回状态，期望值和概率可接入实数自动求导。"""
    def __init__(self, qubits: int, gates: Iterable[Gate] = ()) -> None:
        if not isinstance(qubits, int) or qubits < 1: raise ValueError("量子比特数量必须是正整数")
        self.qubits = qubits
        self.gates: list[Gate] = []
        for gate in gates: self.append(gate)

    def append(self, gate: Gate) -> Self:
        """在电路末尾增加一个门。"""
        selected_wires(self.qubits, gate.wires)
        self.gates.append(gate)
        return self

    def run(self, state: State | None = None) -> State:
        """默认从全零态开始；传入状态时先复制，不修改原状态。"""
        if state is not None and state.qubits != self.qubits: raise ValueError("输入量子态与电路的量子比特数量必须相同")
        result, gates, _, _ = self._snapshot(state)
        for gate in gates: result.apply(gate)
        return result

    def parameters(self) -> tuple[Tensor, ...]:
        """收集门引用的 Tensor 角度，共享参数只返回一次。"""
        return tuple(dict.fromkeys(gate.angle for gate in self.gates if isinstance(gate.angle, Tensor)))

    def _snapshot(self, state: State | None) -> tuple[State, tuple[Gate, ...], tuple[Tensor, ...], tuple[tuple[int, float | Tensor], ...]]:
        """复制初态和门的数值，同时保留每个 Tensor 角度对应的计算关系。"""
        if state is not None and state.qubits != self.qubits: raise ValueError("输入量子态与电路的量子比特数量必须相同")
        parameters = self.parameters()
        if parameters: get_backend(*(parameter.storage for parameter in parameters))
        initial = (State(self.qubits, device=parameters[0].device, dtype="complex64" if parameters[0].dtype == "float32" else "complex128")
                   if parameters and state is None else State(self.qubits) if state is None else state.copy())
        if parameters and (initial.device != parameters[0].device or initial.dtype != ("complex64" if parameters[0].dtype == "float32" else "complex128")):
            raise ValueError("量子初态的设备和精度必须与旋转角度一致")
        gates, parents, rotations = [], [], []
        for position, gate in enumerate(self.gates):
            angle = gate.angle
            if isinstance(angle, Tensor):
                value = angle.item() if angle.device == "cpu" else Tensor(angle.storage.copy(), shape=angle.shape)
                parents.append(angle)
                rotations.append((position, value))
                gates.append(gate.bind(value))
            else:
                gates.append(gate.bind())
        return initial, tuple(gates), tuple(parents), tuple(rotations)

    @overload
    def expectation(self, observable: Operator, state: State | None = None, *, method: str = "adjoint") -> float | Tensor: ...

    @overload
    def expectation(self, observable: Iterable[Operator], state: State | None = None, *, method: str = "adjoint") -> list[float] | Tensor: ...

    def expectation(self, observable: Operator | Iterable[Operator], state: State | None = None, *, method: str = "adjoint") -> float | list[float] | Tensor:
        """共用一次模拟读取观测量，默认伴随求导，也可选择参数移位。"""
        if method not in ("adjoint", "parameter-shift"): raise ValueError("求导方法只支持 adjoint 或 parameter-shift")
        single = isinstance(observable, Operator)
        observables = tuple(as_observable(item) for item in ((observable,) if single else observable))
        if not observables: raise ValueError("至少需要一个观测量")
        initial, gates, parents, rotations = self._snapshot(state)
        result = initial.copy()
        for gate in gates: result.apply(gate)
        values = expectation_values(result.storage, self.qubits, observables)
        if not parents:
            exported = values.tolist()
            return exported[0] if single else exported
        backward = adjoint_backward if method == "adjoint" else parameter_shift_backward
        rule = partial(backward, state=result if method == "adjoint" else initial,
                       gates=gates, observables=observables, rotations=rotations)
        return record(Tensor(values, shape=() if single else (len(observables),)), parents, rule)

    def probabilities(self, wires: int | tuple[int, ...] | None = None, state: State | None = None, *, method: str = "adjoint") -> list[float] | Tensor:
        """读取全部或所选比特的概率；Tensor 角度保留计算关系，顺序与 wires 一致。"""
        if method not in ("adjoint", "parameter-shift"): raise ValueError("求导方法只支持 adjoint 或 parameter-shift")
        wires = selected_wires(self.qubits, wires)
        initial, gates, parents, rotations = self._snapshot(state)
        result = initial.copy()
        for gate in gates: result.apply(gate)
        values = get_backend(result.device).probabilities(result.storage, self.qubits, wires)
        if not parents: return values.tolist()
        backward = adjoint_probabilities_backward if method == "adjoint" else parameter_shift_probabilities_backward
        rule = partial(backward, state=result if method == "adjoint" else initial,
                       gates=gates, wires=wires, rotations=rotations)
        return record(Tensor(values, shape=(2 ** len(wires),)), parents, rule)
