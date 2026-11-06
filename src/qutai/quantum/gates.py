from cmath import exp, isclose
from collections.abc import Callable
from math import cos, pi, sin, sqrt

from ..backends.device import get_backend
from ..backends.storage import Storage
from ..core.tensor import Tensor
from .observables import Hermitian, Matrix, Observable, Operator


class Gate(Operator):
    """小矩阵只描述所选量子比特；wires 的顺序就是矩阵的基底顺序。"""
    def __init__(self, name: str, wires: tuple[int, ...], matrix: Matrix, angle: float | None = None) -> None:
        if isinstance(angle, Tensor): raise TypeError("只有 RX、RY、RZ 旋转门接受 Tensor 角度")
        wires = tuple(wires)
        matrix = tuple(tuple(map(complex, row)) for row in matrix)
        if not wires or any(not isinstance(wire, int) or wire < 0 for wire in wires):
            raise ValueError("量子比特编号必须是非负整数")
        if len(set(wires)) != len(wires): raise ValueError("一个门中的量子比特编号不能重复")
        size = 2 ** len(wires)
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("量子门矩阵的形状必须与所选量子比特数量一致")
        for row in range(size):
            for column in range(size):
                product = sum(matrix[k][row].conjugate() * matrix[k][column] for k in range(size))
                if not isclose(product, float(row == column), abs_tol=1e-10):
                    raise ValueError("量子门矩阵必须是酉矩阵")
        self.name, self.wires = name, wires
        self._matrix = matrix
        self.angle: float | Tensor | None = angle

    @property
    def matrix(self) -> Matrix:
        """固定门的局部矩阵。"""
        return self._matrix

    @property
    def adjoint_matrix(self) -> Matrix:
        """共轭转置矩阵；酉门的共轭转置就是逆矩阵。"""
        return tuple(tuple(value.conjugate() for value in row) for row in zip(*self.matrix, strict=True))

    @property
    def derivative_matrix(self) -> Matrix:
        """只有参数化旋转门提供角度导数，普通固定门不参与参数求导。"""
        raise TypeError("只有参数化旋转门支持角度求导")

    def bind(self, angle: float | None = None) -> Gate:
        """复制固定门的数值快照；只有旋转门可以替换角度。"""
        if angle is not None: raise ValueError("只有旋转门可以替换角度")
        if isinstance(self.angle, Tensor): raise TypeError("只有 RX、RY、RZ 旋转门接受 Tensor 角度")
        return Gate(self.name, self.wires, self.matrix, self.angle)

    def observable(self) -> Observable:
        """固定厄米门也可读取期望值，组合时保存当前比特和矩阵。"""
        if isinstance(self.angle, Tensor): raise TypeError("观测量必须固定，不能使用 Tensor 角度")
        return Hermitian(self.wires, self.matrix)

    def _apply(self, values: Storage, qubits: int, *, adjoint: bool = False, derivative: bool = False) -> Storage:
        """固定门及 CPU 旋转共用原来的局部矩阵路径。"""
        matrix = self.derivative_matrix if derivative else self.adjoint_matrix if adjoint else self.matrix
        return get_backend(values.device).apply_gate(values, qubits, self.wires, matrix)


def _angle(angle: float | Tensor) -> float:
    """读取当前角度；保留门中原来的 Tensor 引用。"""
    return angle.item() if isinstance(angle, Tensor) else angle


class _Rotation(Gate):
    """旋转门保留角度，每次执行重新生成矩阵。"""
    def __init__(self, name: str, wire: int, angle: float | Tensor, operation: Callable[[float], Matrix]) -> None:
        if isinstance(angle, Tensor) and angle.size != 1: raise ValueError("只有包含一个元素的值才能取出单个数字")
        # Tensor 角度在执行时读取；创建门不把 GPU 参数导回 CPU。
        super().__init__(name, (wire,), operation(0.0 if isinstance(angle, Tensor) else angle))
        self.angle: float | Tensor = angle
        self._operation = operation

    @property
    def matrix(self) -> Matrix:
        return self._operation(_angle(self.angle))

    @property
    def derivative_matrix(self) -> Matrix:
        """RX、RY、RZ 的精确导数为旋转角增加 pi 后的矩阵乘以 1/2。"""
        return tuple(tuple(value / 2 for value in row) for row in self._operation(_angle(self.angle) + pi))

    def bind(self, angle: float | Tensor | None = None) -> Gate:
        """生成独立的角度快照，也用于单个门的参数偏移。"""
        value = angle
        if angle is None:
            value = self.angle
            if isinstance(value, Tensor):
                value = value.item() if value.device == "cpu" else Tensor(value.storage.copy(), shape=value.shape)
        return _Rotation(self.name, self.wires[0], value, self._operation)

    def _apply(self, values: Storage, qubits: int, *, adjoint: bool = False, derivative: bool = False) -> Storage:
        """GPU 旋转直接消费设备角度；状态向量和局部导数都留在 GPU。"""
        if isinstance(self.angle, Tensor) and self.angle.device != "cpu":
            if values.device != self.angle.device: raise ValueError("旋转角度与量子态必须位于同一设备")
            return get_backend(values.device).apply_rotation(values, qubits, self.wires[0], self.name,
                                                             self.angle.storage, adjoint=adjoint, derivative=derivative)
        return super()._apply(values, qubits, adjoint=adjoint, derivative=derivative)


def _rx(angle: float) -> Matrix:
    c, s = cos(angle / 2), -1j * sin(angle / 2)
    return ((c, s), (s, c))


def _ry(angle: float) -> Matrix:
    c, s = cos(angle / 2), sin(angle / 2)
    return ((c, -s), (s, c))


def _rz(angle: float) -> Matrix:
    return ((exp(-0.5j * angle), 0), (0, exp(0.5j * angle)))


def I(wire: int) -> Gate:
    """恒等门，不改变量子态。"""
    return Gate("I", (wire,), ((1, 0), (0, 1)))


def H(wire: int) -> Gate:
    """Hadamard 门，将计算基态变为等幅叠加。"""
    scale = 1 / sqrt(2)
    return Gate("H", (wire,), ((scale, scale), (scale, -scale)))


def X(wire: int) -> Gate:
    """Pauli X 门，交换 |0⟩ 与 |1⟩。"""
    return Gate("X", (wire,), ((0, 1), (1, 0)))


def Y(wire: int) -> Gate:
    """Pauli Y 门，翻转比特并改变相位。"""
    return Gate("Y", (wire,), ((0, -1j), (1j, 0)))


def Z(wire: int) -> Gate:
    """Pauli Z 门，将 |1⟩ 的振幅取负。"""
    return Gate("Z", (wire,), ((1, 0), (0, -1)))


def S(wire: int) -> Gate:
    """S 相位门，将 |1⟩ 的振幅乘以 i。"""
    return Gate("S", (wire,), ((1, 0), (0, 1j)))


def T(wire: int) -> Gate:
    """T 相位门，将 |1⟩ 的振幅乘以 exp(iπ/4)。"""
    return Gate("T", (wire,), ((1, 0), (0, exp(1j * pi / 4))))


def RX(wire: int, angle: float | Tensor) -> Gate:
    """绕 X 轴旋转 angle 弧度。"""
    return _Rotation("RX", wire, angle, _rx)


def RY(wire: int, angle: float | Tensor) -> Gate:
    """绕 Y 轴旋转 angle 弧度。"""
    return _Rotation("RY", wire, angle, _ry)


def RZ(wire: int, angle: float | Tensor) -> Gate:
    """绕 Z 轴旋转 angle 弧度。"""
    return _Rotation("RZ", wire, angle, _rz)


def CNOT(control: int, target: int) -> Gate:
    """控制比特为 1 时翻转目标比特。"""
    return Gate("CNOT", (control, target), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)))


def CZ(control: int, target: int) -> Gate:
    """两个所选比特都是 1 时将振幅取负。"""
    return Gate("CZ", (control, target), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1)))


def SWAP(first: int, second: int) -> Gate:
    """交换两个量子比特。"""
    return Gate("SWAP", (first, second), ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
