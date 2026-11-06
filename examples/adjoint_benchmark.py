"""同一电路比较伴随与参数移位的梯度，以及一次前向加反向的耗时。"""

import platform
import sys
from random import Random
from statistics import median
from time import perf_counter
from typing import cast

from qutai import Tensor
from qutai.quantum import CNOT, H, RX, RY, RZ, X, Y, Z, Circuit, Observable


def build_case(qubits: int, depth: int) -> tuple[Circuit, Tensor, Observable]:
    """固定数据与门结构；仅在计时前构造一次，两个方法共享相同输入。"""
    random = Random(42)
    weight = Tensor([random.uniform(-0.8, 0.8) for _ in range(qubits * depth)], shape=(depth, qubits), requires_grad=True)
    gates = [H(wire) for wire in range(qubits)]
    rotations = (RX, RY, RZ)
    for layer in range(depth):
        gates.extend(rotations[(layer + wire) % 3](wire, weight[layer, wire]) for wire in range(qubits))
        gates.extend(CNOT(wire, wire + 1) for wire in range(qubits - 1))
    cost = 0.7 * Z(0) - 0.2 * X(qubits - 1) + 0.3 * (Y(0) @ Y(qubits - 1))
    return Circuit(qubits, gates), weight, cost


def measure(circuit: Circuit, weight: Tensor, cost: Observable, method: str) -> tuple[float, list[float]]:
    """计时只包含求期望与 backward；清梯度和读取结果均在计时之外。"""
    weight.zero_grad()
    start = perf_counter()
    loss = cast(Tensor, circuit.expectation(cost, method=method))
    loss.backward()
    elapsed = perf_counter() - start
    return elapsed, cast(Tensor, weight.grad).storage.tolist()


def run_case(qubits: int, depth: int, repeats: int = 3) -> dict[str, int | float]:
    """各预热一次，再交替测量顺序，使用中位数减小偶发波动。"""
    circuit, weight, cost = build_case(qubits, depth)
    methods = ("adjoint", "parameter-shift")
    warmup = [measure(circuit, weight, cost, method)[1] for method in methods]
    error = max(abs(left - right) for left, right in zip(*warmup, strict=True))
    if error > 1e-8: raise AssertionError(f"两种方法的梯度不一致：最大误差 {error:.3e}")
    samples = {method: [] for method in methods}
    for repeat in range(repeats):
        for method in methods if repeat % 2 == 0 else reversed(methods):
            elapsed, _ = measure(circuit, weight, cost, method)
            samples[method].append(elapsed)
    adjoint, shift = (median(samples[method]) for method in methods)
    return {
        "qubits": qubits, "depth": depth, "gates": len(circuit.gates),
        "rotations": weight.size, "repeats": repeats,
        "adjoint_seconds": adjoint, "parameter_shift_seconds": shift,
        "speedup": shift / adjoint, "max_gradient_error": error,
    }


def main() -> None:
    print(f"Python {sys.version.split()[0]} · {platform.system()} {platform.machine()}")
    print("同一电路与参数；各预热一次、各测三次；记录前向 + 反向的中位耗时。")
    print("比特 深度 门数 旋转门   伴随(ms) 参数移位(ms)   加速比   最大梯度误差")
    for qubits, depth in ((4, 2), (6, 4), (6, 8)):
        result = run_case(qubits, depth)
        print(f"{qubits:4d} {depth:4d} {result['gates']:4d} {result['rotations']:6d} "
              f"{result['adjoint_seconds'] * 1000:10.3f} {result['parameter_shift_seconds'] * 1000:12.3f} "
              f"{result['speedup']:8.2f}x {result['max_gradient_error']:14.3e}")


if __name__ == "__main__":
    main()
