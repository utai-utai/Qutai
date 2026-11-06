from cmath import isclose
from collections.abc import Iterable

from ..backends.device import get_backend
from ..backends.storage import Storage


type Matrix = tuple[tuple[complex, ...], ...]
type Factor = tuple[tuple[int, ...], Matrix]
type Term = tuple[float, tuple[Factor, ...]]


class Operator:
    """门与观测量共用的组合语法；组合只接受固定的厄米算符。"""
    def observable(self) -> Observable:
        raise NotImplementedError

    def __add__(self, other: Operator | float) -> Observable:
        if not isinstance(other, (Operator, int, float)): return NotImplemented
        return Observable(as_observable(self).terms + _terms(other))

    def __radd__(self, other: float) -> Observable:
        return self + other

    def __neg__(self) -> Observable:
        return self * -1

    def __sub__(self, other: Operator | float) -> Observable:
        if not isinstance(other, (Operator, int, float)): return NotImplemented
        return self + (-as_observable(other) if isinstance(other, Operator) else -other)

    def __rsub__(self, other: float) -> Observable:
        return -self + other

    def __mul__(self, coefficient: float) -> Observable:
        if not isinstance(coefficient, (int, float)): return NotImplemented
        return Observable(tuple((coefficient * weight, factors) for weight, factors in as_observable(self).terms))

    def __rmul__(self, coefficient: float) -> Observable:
        return self * coefficient

    def __truediv__(self, coefficient: float) -> Observable:
        if not isinstance(coefficient, (int, float)): return NotImplemented
        return self * (1 / coefficient)

    def __matmul__(self, other: Operator) -> Observable:
        if not isinstance(other, Operator): return NotImplemented
        terms = []
        for left_weight, left in as_observable(self).terms:
            for right_weight, right in as_observable(other).terms:
                left_wires = {wire for wires, _ in left for wire in wires}
                right_wires = {wire for wires, _ in right for wire in wires}
                if left_wires & right_wires: raise ValueError("张量积的各项必须作用于不同的量子比特")
                terms.append((left_weight * right_weight, left + right))
        return Observable(tuple(terms))


class Observable(Operator):
    """实系数局部厄米算符乘积的和；terms 是内部表示，使用 Hermitian、+、*、@ 构造。"""
    def __init__(self, terms: Iterable[Term]) -> None:
        self.terms = tuple(terms)

    def bind(self) -> Observable:
        """复制当前观测量；各项的比特和矩阵均已保存为不可变元组。"""
        return Observable(self.terms)

    def observable(self) -> Observable:
        return self.bind()

    def apply(self, amplitudes: Storage | Iterable[complex], qubits: int) -> Storage:
        """逐项作用于振幅，不构造整个系统的稠密矩阵，也不修改输入。"""
        backend = get_backend(amplitudes.device if isinstance(amplitudes, Storage) else "cpu")
        if not isinstance(amplitudes, Storage): amplitudes = backend.storage(amplitudes, dtype="complex128")
        result = backend.full(amplitudes.size, 0j, dtype=amplitudes.dtype)
        for coefficient, factors in self.terms:
            values = amplitudes
            for wires, matrix in factors:
                if any(wire >= qubits for wire in wires): raise ValueError("观测量的量子比特编号超出状态范围")
                values = backend.apply_gate(values, qubits, wires, matrix)
            result = backend.add(result, backend.scale(values, coefficient))
        return result


def expectation_values(amplitudes: Storage, qubits: int, observables: tuple[Observable, ...]) -> Storage:
    """将各观测量的期望保留为实数存储，供电路与求导共用。"""
    backend = get_backend(amplitudes.device)
    values = tuple(backend.real_inner(amplitudes, observable.apply(amplitudes, qubits)) for observable in observables)
    return backend.concatenate(values, ((1,),) * len(values))


def _terms(value: Operator | float) -> tuple[Term, ...]:
    """普通数字表示相应倍数的恒等算符。"""
    return ((float(value), ()),) if isinstance(value, (int, float)) else as_observable(value).terms


def as_observable(operator: Operator) -> Observable:
    """将门或组合观测量转换为固定的数值快照。"""
    if not isinstance(operator, Operator): raise TypeError("观测量必须是固定的厄米算符")
    return operator.observable()


def Hermitian(wires: tuple[int, ...], matrix: Matrix) -> Observable:
    """用局部厄米矩阵构造观测量；矩阵不需要是酉矩阵。"""
    wires = tuple(wires)
    matrix = tuple(tuple(map(complex, row)) for row in matrix)
    if not wires or any(not isinstance(wire, int) or wire < 0 for wire in wires) or len(set(wires)) != len(wires):
        raise ValueError("观测量的量子比特编号必须是不同的非负整数")
    size = 2 ** len(wires)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("观测量矩阵的形状必须与所选量子比特数量一致")
    if any(not isclose(value, matrix[column][row].conjugate(), abs_tol=1e-10)
           for row, values in enumerate(matrix) for column, value in enumerate(values)):
        raise ValueError("期望值的观测量必须是厄米矩阵")
    return Observable(((1.0, ((wires, matrix),)),))
