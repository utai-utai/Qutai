import argparse
import json
from math import cos, isfinite, log10, pi, prod, sin, sqrt
from pathlib import Path
from random import Random
from statistics import mean, variance

from qutai import Tensor, qnn, quantum, visualization


def global_z(qubits):
    operator = quantum.Z(0)
    for wire in range(1, qubits):
        operator = operator @ quantum.Z(wire)
    return operator


def gradient_pair(qubits, depth, seed, *, initialization="wide", kind="pqc",
                  device="cuda", dtype="float32", method="adjoint"):
    """同一参数点分别求局部/全局 cost 对第一个角度的梯度。"""
    if initialization not in ("wide", "small"):
        raise ValueError("初始化只能是 wide 或 small")
    random = Random(seed)
    spread = pi if initialization == "wide" else 0.1
    count = qubits * depth if kind == "pqc" else qubits
    angles = [random.uniform(-spread, spread) for _ in range(count)]
    if kind == "pqc":
        layer = qnn.PQC(qubits, depth=depth, entangler="cz", seed=23)
        # seed=23 固定结构，第一门是 RY；只重采样角度，不重采样电路。
        if layer.rotations[0][0] != "RY":
            raise AssertionError("选中参数的旋转轴发生变化")
        layer.weight = Tensor(angles, shape=(depth, qubits), requires_grad=True)
        layer.to(device, dtype=dtype)
        parameter = layer.weight
        circuit = layer.circuit([0.0] * qubits)
    elif kind == "factorized":
        if depth != 1:
            raise ValueError("解析乘积态对照只有一层RY")
        parameter = Tensor(angles, device=device, dtype=dtype, requires_grad=True)
        circuit = quantum.Circuit(qubits, [quantum.RY(wire, parameter[wire]) for wire in range(qubits)])
    else:
        raise ValueError("kind 只能是 pqc 或 factorized")
    values = circuit.expectation([quantum.Z(0), global_z(qubits)], method=method)
    costs = values.tolist()
    gradients = []
    for index in range(2):
        parameter.zero_grad()
        values[index].backward(retain_graph=index == 0)
        gradients.append(parameter.grad.storage.tolist()[0])
    if any(not isfinite(value) or abs(value) > 1.0001 for value in (*costs, *gradients)):
        raise ArithmeticError("归一化 Pauli cost 或梯度超出数值范围")
    result = dict(kind=kind, initialization=initialization, qubits=qubits, depth=depth,
                  seed=seed, local_cost=costs[0], global_cost=costs[1],
                  local_gradient=gradients[0], global_gradient=gradients[1])
    if kind == "factorized":
        expected = [-sin(angles[0]), -sin(angles[0]) * prod(cos(value) for value in angles[1:])]
        result["analytic_gradient_max_abs"] = max(abs(a - b) for a, b in zip(gradients, expected))
    return result


def collect(qubits=(2, 4, 6, 8), depths=(1, 4), samples=24, *, device="cuda", dtype="float32"):
    """每组共享 seed，局部/全局 cost 共用电路和初始参数。"""
    if samples < 4 or any(n < 1 for n in qubits) or any(d < 1 for d in depths):
        raise ValueError("samples 至少为4，qubits/depth 必须为正数")
    records = []
    for n in qubits:
        for depth in depths:
            for initialization in ("wide", "small"):
                for seed in range(samples):
                    records.append(gradient_pair(n, depth, 1000 + seed, initialization=initialization,
                                                 device=device, dtype=dtype))
        for seed in range(samples):
            records.append(gradient_pair(n, 1, 1000 + seed, kind="factorized", device=device, dtype=dtype))
    return records


def bootstrap_variance(values, *, seed=43, repeats=400):
    """参数初始化的非参数 percentile bootstrap；不是硬件 shot 置信区间。"""
    random = Random(seed)
    estimates = sorted(variance(random.choices(values, k=len(values))) for _ in range(repeats))
    return estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]


def summarize(records):
    groups = {}
    for row in records:
        for cost in ("local", "global"):
            key = row["kind"], row["initialization"], row["qubits"], row["depth"], cost
            groups.setdefault(key, []).append(row[cost + "_gradient"])
    rows = []
    for (kind, initialization, n, depth, cost), values in groups.items():
        low, high = bootstrap_variance(values)
        rows.append(dict(kind=kind, initialization=initialization, qubits=n, depth=depth, cost=cost,
                         samples=len(values), mean=mean(values), rms=sqrt(mean(v * v for v in values)),
                         variance=variance(values), ci_low=low, ci_high=high,
                         near_zero_fraction=sum(abs(v) < 1e-3 for v in values) / len(values)))
    return rows


def variance_plot(rows):
    chart = visualization.Plot(title="固定 PQC：初始化梯度方差", xlabel="量子比特数", ylabel="log10(样本方差)")
    depths = sorted({row["depth"] for row in rows if row["kind"] == "pqc"})
    for depth in depths:
        for cost in ("local", "global"):
            selected = sorted((r for r in rows if r["kind"] == "pqc" and r["initialization"] == "wide"
                               and r["depth"] == depth and r["cost"] == cost), key=lambda r: r["qubits"])
            chart.line([r["qubits"] for r in selected], [log10(max(r["variance"], 1e-30)) for r in selected],
                       label=f"{cost}, depth={depth}")
    return chart


def main():
    parser = argparse.ArgumentParser(description="PQC 初始化梯度统计")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = collect(args.qubits, args.depths, args.samples, device=args.device, dtype=args.dtype)
    result = dict(device=args.device, dtype=args.dtype, qubits=args.qubits, depths=args.depths,
                  samples_per_group=args.samples, selected_parameter="theta[0,0]", structure_seed=23,
                  records=records, summary=summarize(records))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
