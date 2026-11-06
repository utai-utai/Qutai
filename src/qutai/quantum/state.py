from collections import Counter
from copy import copy as copy_object
from collections.abc import Iterable
from math import isclose
from random import Random
from typing import Self, overload

from ..backends.device import get_backend, transfer
from ..backends.storage import Storage
from .gates import Gate
from .observables import Operator, as_observable, expectation_values


def selected_wires(qubits: int, wires: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """检查比特编号并保留调用者指定的顺序；0 是最左侧比特。"""
    result = tuple(range(qubits)) if wires is None else (wires,) if isinstance(wires, int) else tuple(wires)
    if not result or any(not isinstance(wire, int) or not 0 <= wire < qubits for wire in result):
        raise ValueError("量子比特编号必须是有效范围内的整数，且至少选择一个比特")
    if len(set(result)) != len(result): raise ValueError("所选量子比特编号不能重复")
    return result


class State:
    """纯量子态；振幅按 |00…0⟩、|00…1⟩……排序，编号 0 为最高位。"""
    def __init__(self, qubits: int, amplitudes: Iterable[complex] | Storage | None = None, *, device: str | None = None, dtype: str | None = None) -> None:
        if not isinstance(qubits, int) or qubits < 1: raise ValueError("量子比特数量必须是正整数")
        self.qubits = qubits
        device = device if device is not None else amplitudes.device if isinstance(amplitudes, Storage) else "cpu"
        dtype = dtype if dtype is not None else amplitudes.dtype if isinstance(amplitudes, Storage) else "complex128"
        if dtype not in ("complex64", "complex128"): raise ValueError("量子振幅只支持 complex64 或 complex128")
        backend = get_backend(device)
        if isinstance(amplitudes, Storage) and amplitudes.device != device: raise ValueError("输入振幅与目标设备必须相同")
        self.storage = backend.basis_state(qubits, dtype=dtype) if amplitudes is None else backend.from_storage(amplitudes, dtype=dtype) if isinstance(amplitudes, Storage) else backend.storage(amplitudes, dtype=dtype)
        if self.storage.size != 2 ** qubits: raise ValueError("振幅数量必须等于 2 的量子比特数量次方")
        if not isclose(backend.inner(self.storage, self.storage).real, 1.0, rel_tol=1e-6 if dtype == "complex64" else 1e-10, abs_tol=1e-7 if dtype == "complex64" else 1e-12):
            raise ValueError("量子态必须归一化：振幅绝对值的平方和应为 1")

    @property
    def device(self) -> str:
        return self.storage.device

    @property
    def dtype(self) -> str:
        return self.storage.dtype

    @property
    def amplitudes(self) -> Storage:
        """CPU 振幅的实时调试入口；计算过程统一使用 storage。"""
        if self.device != "cpu": raise ValueError("只有 CPU 状态支持直接读取振幅存储")
        return self.storage

    @amplitudes.setter
    def amplitudes(self, values: Iterable[complex] | Storage) -> None:
        self.storage = get_backend(self.device).storage(values, dtype=self.dtype)

    def to(self, device: str | None = None, *, dtype: str | None = None, copy: bool = False) -> State:
        """转换设备或复数精度；相同配置默认返回自身，不改变原状态。"""
        if dtype is not None and dtype not in ("complex64", "complex128"): raise ValueError("量子振幅只支持 complex64 或 complex128")
        values = transfer(self.storage, device, dtype, copy=copy)
        if values is self.storage: return self
        result = copy_object(self)
        result.storage = values
        return result

    def copy(self) -> State:
        """复制状态，后续门操作和测量互不影响。"""
        return self.to(copy=True)

    def apply(self, gate: Gate) -> Self:
        """原地执行一个门，只计算局部振幅，不构造完整的大矩阵。"""
        selected_wires(self.qubits, gate.wires)
        self.storage = get_backend(self.device).apply_gate(self.storage, self.qubits, gate.wires, gate.matrix)
        return self

    def probabilities(self, wires: int | tuple[int, ...] | None = None) -> list[float]:
        """返回计算基测量概率；指定部分比特时，对其余比特求和。"""
        wires = selected_wires(self.qubits, wires)
        return get_backend(self.device).probabilities(self.storage, self.qubits, wires).tolist()

    @overload
    def expectation(self, observable: Operator) -> float: ...

    @overload
    def expectation(self, observable: Iterable[Operator]) -> list[float]: ...

    def expectation(self, observable: Operator | Iterable[Operator]) -> float | list[float]:
        """读取一个或多个固定观测量的精确期望，不改变状态或建立计算图。"""
        single = isinstance(observable, Operator)
        observables = tuple(as_observable(item) for item in ((observable,) if single else observable))
        if not observables: raise ValueError("至少需要一个观测量")
        values = expectation_values(self.storage, self.qubits, observables).tolist()
        return values[0] if single else values

    def sample(self, shots: int = 1, *, wires: int | tuple[int, ...] | None = None, seed: int | None = None) -> list[str]:
        """从同一状态独立采样多次，不改变当前状态。"""
        if not isinstance(shots, int) or shots < 0: raise ValueError("采样次数必须是非负整数")
        wires = selected_wires(self.qubits, wires)
        probabilities = self.probabilities(wires)
        samples = Random(seed).choices(range(len(probabilities)), weights=probabilities, k=shots)
        return [format(index, f"0{len(wires)}b") for index in samples]

    def counts(self, shots: int = 1024, *, wires: int | tuple[int, ...] | None = None, seed: int | None = None) -> dict[str, int]:
        """统计独立采样结果，只返回实际出现过的位串。"""
        return dict(sorted(Counter(self.sample(shots, wires=wires, seed=seed)).items()))

    def measure(self, wires: int | tuple[int, ...] | None = None, *, seed: int | None = None) -> str:
        """测量一次并原地坍缩；未测比特保留与该结果相容的叠加。"""
        wires = selected_wires(self.qubits, wires)
        probabilities = self.probabilities(wires)
        outcome = Random(seed).choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        self.storage = get_backend(self.device).collapse(self.storage, self.qubits, wires, outcome, probabilities[outcome])
        return format(outcome, f"0{len(wires)}b")

    def __repr__(self) -> str:
        return f"State({self.qubits}, amplitudes={self.storage!r})"
