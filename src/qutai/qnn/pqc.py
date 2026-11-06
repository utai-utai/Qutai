from math import pi
from random import Random
from typing import cast

from ..core.context import set_recording
from ..core.migration import move
from ..core.tensor import Tensor, stack
from ..core.types import Data
from ..quantum.circuit import Circuit
from ..quantum.gates import CNOT, CZ, RX, RY, RZ, Z


_ROTATIONS = {"RX": RX, "RY": RY, "RZ": RZ}


class PQC:
    """固定随机旋转轴的参数化电路，将 (..., qubits) 输入变为同形状实数特征。"""
    def __init__(self, qubits: int, depth: int = 2, *, entangler: str = "cnot", seed: int | None = None) -> None:
        if type(qubits) is not int or qubits <= 0: raise ValueError("量子比特数量必须是正整数")
        if type(depth) is not int or depth <= 0: raise ValueError("电路深度必须是正整数")
        entangler = entangler.lower()
        if entangler not in ("cnot", "cz"): raise ValueError("纠缠门目前只支持 CNOT 和 CZ")
        self.qubits, self.depth, self.entangler = qubits, depth, entangler
        random = Random(seed)
        self.rotations = tuple(tuple(random.choice(tuple(_ROTATIONS)) for _ in range(qubits)) for _ in range(depth))
        self.weight = Tensor([random.uniform(-pi, pi) for _ in range(depth * qubits)], shape=(depth, qubits), requires_grad=True)

    def to(self, device: str | None = None, *, dtype: str | None = None) -> PQC:
        """原地迁移模型参数、已有梯度与子层的运行统计量。"""
        return move(self, device, dtype)

    def parameters(self) -> tuple[Tensor, ...]:
        """只返回层自己的角度权重，输入数据不算模型参数。"""
        return (self.weight,)

    def circuit(self, data: Data | Tensor) -> Circuit:
        """为单个样本建立电路，可直接交给电路可视化；不执行模拟。"""
        tensor = data if isinstance(data, Tensor) else Tensor(data, device=self.weight.device, dtype=self.weight.dtype)
        if tensor.shape != (self.qubits,): raise ValueError("单个样本的特征数量必须等于量子比特数量")
        gates = [RY(wire, tensor[wire]) for wire in range(self.qubits)]
        entangle = CNOT if self.entangler == "cnot" else CZ
        for layer, rotations in enumerate(self.rotations):
            gates.extend(_ROTATIONS[name](wire, self.weight[layer, wire]) for wire, name in enumerate(rotations))
            gates.extend(entangle(wire, wire + 1) for wire in range(self.qubits - 1))
        # 固定读出旋转让 Z 读出同时包含 X 信息，避免末尾 CZ 对读出完全不可见。
        gates.extend(RY(wire, pi / 4) for wire in range(self.qubits))
        return Circuit(self.qubits, gates)

    def __call__(self, data: Data | Tensor, *, train: bool = True) -> Tensor:
        """每个样本独立准备量子态，各样本共享权重；输入与权重都保留梯度。"""
        tensor = data if isinstance(data, Tensor) else Tensor(data, device=self.weight.device, dtype=self.weight.dtype)
        if not tensor.shape or tensor.shape[-1] != self.qubits: raise ValueError("输入的最后一维长度必须等于量子比特数量")
        with set_recording(train):
            if tensor.size == 0: return tensor + self.weight.sum() * 0
            rows = tensor.reshape((-1, self.qubits))
            outputs = []
            for index in range(len(rows)):
                circuit = self.circuit(rows[index])
                # 所有比特共用一次模拟；Tensor 角度使整组读出保留计算关系。
                outputs.append(cast(Tensor, circuit.expectation([Z(wire) for wire in range(self.qubits)])))
            return stack(outputs).reshape(tensor.shape)
