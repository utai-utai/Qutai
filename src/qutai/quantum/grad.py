from math import pi

from ..backends.device import get_backend
from ..backends.storage import Storage
from ..core.tensor import Tensor
from .gates import Gate
from .observables import Observable, expectation_values
from .state import State


def _shifted_state(state: State, gates: tuple[Gate, ...], position: int, angle: float | Tensor) -> State:
    """只偏移指定位置的门，所有读出共用偏移后的同一个态。"""
    result = state.copy()
    for index, gate in enumerate(gates):
        result.apply(gate.bind(angle) if index == position else gate)
    return result


def parameter_shift_backward(gradient: Storage, *, state: State, gates: tuple[Gate, ...], observables: tuple[Observable, ...], rotations: tuple[tuple[int, float | Tensor], ...]) -> tuple[Storage, ...]:
    """逐门偏移后汇总读出；实数梯度一直保留在后端存储中。"""
    backend = get_backend(state.device)
    contributions = []
    for position, angle in rotations:
        upper = _shifted_state(state, gates, position, angle + pi / 2)
        lower = _shifted_state(state, gates, position, angle - pi / 2)
        difference = backend.subtract(expectation_values(upper.storage, state.qubits, observables),
                                      expectation_values(lower.storage, state.qubits, observables))
        contributions.append(backend.real_inner(gradient, difference, scale=0.5))
    return tuple(contributions)


def parameter_shift_probabilities_backward(gradient: Storage, *, state: State, gates: tuple[Gate, ...], wires: tuple[int, ...], rotations: tuple[tuple[int, float | Tensor], ...]) -> tuple[Storage, ...]:
    """每次偏移读取整组边缘概率，再按上游梯度加权汇总。"""
    backend = get_backend(state.device)
    contributions = []
    for position, angle in rotations:
        upper = _shifted_state(state, gates, position, angle + pi / 2)
        lower = _shifted_state(state, gates, position, angle - pi / 2)
        difference = backend.subtract(backend.probabilities(upper.storage, state.qubits, wires),
                                      backend.probabilities(lower.storage, state.qubits, wires))
        contributions.append(backend.real_inner(gradient, difference, scale=0.5))
    return tuple(contributions)


def _adjoint(bra: Storage, *, state: State, gates: tuple[Gate, ...], rotations: tuple[tuple[int, float | Tensor], ...]) -> tuple[Storage, ...]:
    """逆序撤销门；伴随向量允许未归一化，直接使用复数存储。"""
    backend = get_backend(state.device)
    ket = state.storage.copy()
    positions = {position: index for index, (position, _) in enumerate(rotations)}
    contributions = [backend.full(1, dtype="float32" if state.dtype == "complex64" else "float64") for _ in rotations]
    for position in reversed(range(len(gates))):
        gate = gates[position]
        previous = gate._apply(ket, state.qubits, adjoint=True)
        if position in positions:
            derivative = gate._apply(previous, state.qubits, derivative=True)
            contributions[positions[position]] = backend.real_inner(bra, derivative, scale=2)
        bra = gate._apply(bra, state.qubits, adjoint=True)
        ket = previous
    return tuple(contributions)


def adjoint_backward(gradient: Storage, *, state: State, gates: tuple[Gate, ...], observables: tuple[Observable, ...], rotations: tuple[tuple[int, float | Tensor], ...]) -> tuple[Storage, ...]:
    """合并所有观测量的上游梯度，再逆序计算旋转角度的梯度。"""
    values = tuple(observable.apply(state.storage, state.qubits) for observable in observables)
    bra = get_backend(state.device).weighted_sum(values, gradient)
    return _adjoint(bra, state=state, gates=gates, rotations=rotations)


def adjoint_probabilities_backward(gradient: Storage, *, state: State, gates: tuple[Gate, ...], wires: tuple[int, ...], rotations: tuple[tuple[int, float | Tensor], ...]) -> tuple[Storage, ...]:
    """后端按所选比特映射概率梯度，无需构造投影矩阵。"""
    bra = get_backend(state.device).probabilities_backward(gradient, state.storage, state.qubits, wires)
    return _adjoint(bra, state=state, gates=gates, rotations=rotations)
