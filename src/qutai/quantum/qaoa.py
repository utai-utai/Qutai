from collections.abc import Iterable
from math import pi
from random import Random
from typing import cast

from ..core.tensor import Tensor
from ..core.migration import move
from .circuit import Circuit
from .gates import CNOT, H, RX, RZ, Z
from .observables import Observable, Operator, as_observable
from .state import State


def maxcut(edges: Iterable[tuple[int, int]]) -> Observable:
    """每条边连接不同分组时能量为 -1；最小化期望值即最大化切边数。"""
    return sum(((Z(first) @ Z(second) - 1) / 2 for first, second in edges), Observable(()))


def _z_terms(cost: Observable, qubits: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
    """提取带系数的 Z 字符串；恒等项只改变整体相位，可省略演化。"""
    terms = []
    for coefficient, factors in cost.terms:
        wires = []
        for selected, matrix in factors:
            if any(wire >= qubits for wire in selected): raise ValueError("成本观测量的量子比特编号超出范围")
            if len(selected) != 1 or matrix not in (((1, 0), (0, 1)), ((1, 0), (0, -1))):
                raise ValueError("QAOA 成本目前只支持 I、Z 的张量积与实数加权和")
            if matrix[1][1] == -1: wires.extend(selected)
        if wires and coefficient: terms.append((coefficient, tuple(wires)))
    return tuple(terms)


class QAOA:
    """从等幅叠加态开始，交替执行 Z 成本演化与 X 混合演化。"""
    def __init__(self, qubits: int, cost: Operator, depth: int = 1, *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        if type(qubits) is not int or qubits <= 0: raise ValueError("量子比特数量必须是正整数")
        if type(depth) is not int or depth <= 0: raise ValueError("电路深度必须是正整数")
        self.qubits, self.depth = qubits, depth
        self._cost = as_observable(cost)
        self._terms = _z_terms(self._cost, qubits)
        random = Random(seed)
        self.weight = Tensor([random.uniform(-pi, pi) for _ in range(depth * 2)], shape=(depth, 2), requires_grad=True, device=device, dtype=dtype)

    @property
    def cost(self) -> Observable:
        """读取固定成本的副本，避免外部修改导致成本与电路结构不一致。"""
        return self._cost.bind()

    def to(self, device: str | None = None, *, dtype: str | None = None) -> QAOA:
        """原地迁移可训练角度及其已有梯度。"""
        return move(self, device, dtype)

    def parameters(self) -> tuple[Tensor, ...]:
        """每层一对 gamma、beta，作为同一组参数交给 SGD。"""
        return (self.weight,)

    def circuit(self) -> Circuit:
        """使用当前参数重新建立计算关系；RZ、RX 的角度含旋转约定中的两倍系数。"""
        gates = [H(wire) for wire in range(self.qubits)]
        for layer in range(self.depth):
            gamma, beta = self.weight[layer, 0], self.weight[layer, 1]
            for coefficient, wires in self._terms:
                target = wires[-1]
                gates.extend(CNOT(wire, target) for wire in wires[:-1])
                gates.append(RZ(target, 2 * coefficient * gamma))
                gates.extend(CNOT(wire, target) for wire in reversed(wires[:-1]))
            gates.extend(RX(wire, 2 * beta) for wire in range(self.qubits))
        return Circuit(self.qubits, gates)

    def expectation(self) -> Tensor:
        """返回可求导的成本期望值；复用电路的量子反向规则。"""
        return cast(Tensor, self.circuit().expectation(self._cost))

    def run(self) -> State:
        """用当前参数执行数值模拟，可继续读取概率、采样或画图。"""
        return self.circuit().run()
