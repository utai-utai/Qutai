from html import escape
from math import cos, pi, sin, sqrt
from pathlib import Path

from ..quantum import Circuit as QuantumCircuit, Gate, State, X, Y, Z
from ..quantum.state import selected_wires
from ._svg import save_svg, svg, text


def _columns(gates: tuple[Gate, ...], qubits: int) -> list[int]:
    """独立的门共用一列，多比特门为连线经过的所有比特留出空间。"""
    available = [0] * qubits
    columns = []
    for gate in gates:
        first, last = min(gate.wires), max(gate.wires)
        column = max(available[first:last + 1])
        columns.append(column)
        available[first:last + 1] = [column + 1] * (last - first + 1)
    return columns


class Circuit:
    """只读取电路结构；时间从左到右，q0 是最上方的量子比特。"""
    def __init__(self, circuit: QuantumCircuit, title: str = "") -> None:
        self.qubits, self.gates = circuit.qubits, tuple(gate.bind() for gate in circuit.gates)
        self.title = title

    def _repr_svg_(self) -> str:
        columns = _columns(self.gates, self.qubits)
        width, height = max(360, 190 + (max(columns, default=-1) + 1) * 90), 110 + self.qubits * 70
        body = text(width / 2, 28, self.title, anchor="middle", size=18)
        for wire in range(self.qubits):
            y = 85 + wire * 70
            body += text(24, y + 4, f"q{wire}", size=14)
            body += f'<path data-wire="{wire}" d="M65,{y}H{width - 30}" stroke="#94a3b8" stroke-width="1.6"/>'
        for column, gate in zip(columns, self.gates, strict=True):
            x = 115 + column * 90
            positions = [85 + wire * 70 for wire in gate.wires]
            body += f'<g data-gate="{escape(gate.name, quote=True)}" data-wires="{",".join(map(str, gate.wires))}" data-column="{column}">'
            if len(positions) > 1:
                body += f'<path d="M{x},{min(positions)}V{max(positions)}" stroke="#2563eb" stroke-width="2"/>'
            if len(positions) == 2 and gate.name in ("CNOT", "CZ"):
                body += f'<circle data-role="control" cx="{x}" cy="{positions[0]}" r="5" fill="#2563eb"/>'
                if gate.name == "CZ":
                    body += f'<circle data-role="control" cx="{x}" cy="{positions[1]}" r="5" fill="#2563eb"/>'
                else:
                    y = positions[1]
                    body += f'<circle data-role="target" cx="{x}" cy="{y}" r="13" fill="white" stroke="#2563eb" stroke-width="2"/>'
                    body += f'<path d="M{x - 13},{y}H{x + 13} M{x},{y - 13}V{y + 13}" stroke="#2563eb" stroke-width="2"/>'
            elif len(positions) == 2 and gate.name == "SWAP":
                for y in positions:
                    body += f'<path data-role="swap" d="M{x - 8},{y - 8}L{x + 8},{y + 8} M{x - 8},{y + 8}L{x + 8},{y - 8}" stroke="#2563eb" stroke-width="2"/>'
            else:
                for index, y in enumerate(positions):
                    body += f'<rect x="{x - 33}" y="{y - 23}" width="66" height="46" rx="6" fill="#eef2ff" stroke="#818cf8"/>'
                    detail = f"{gate.angle:.3g}" if gate.angle is not None else f"[{index}]" if len(positions) > 1 else ""
                    body += text(x, y - 3 if detail else y + 5, gate.name, anchor="middle", size=14)
                    if detail: body += text(x, y + 15, detail, anchor="middle", size=11, fill="#64748b")
            body += '</g>'
        body += text(24, height - 18, "从左到右执行；q0 对应最左侧比特。", size=11, fill="#64748b")
        return svg(width, height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())


class Probabilities:
    """显示完整状态或指定比特的边缘概率；读取时保存快照，不改变量子态。"""
    def __init__(self, state: State, *, wires: int | tuple[int, ...] | None = None, title: str = "") -> None:
        self.wires = selected_wires(state.qubits, wires)
        self.values = state.probabilities(self.wires)
        self.qubits = len(self.wires)
        self.title = title

    def _repr_svg_(self) -> str:
        left, top, chart_height = 65, 60, 240
        column = max(48, self.qubits * 9 + 22)
        chart_width = max(270, len(self.values) * column)
        width, height = left + chart_width + 25, 370
        body = text(width / 2, 28, self.title, anchor="middle", size=18)
        for tick in range(5):
            y = top + chart_height * (1 - tick / 4)
            body += f'<path d="M{left},{y:g}H{left + chart_width}" stroke="#e2e8f0"/>'
            body += text(left - 10, y + 4, f"{tick * 25}%", anchor="end", size=11)
        body += f'<path d="M{left},{top}V{top + chart_height}H{left + chart_width}" fill="none" stroke="#64748b"/>'
        step = chart_width / len(self.values)
        for index, value in enumerate(self.values):
            basis = format(index, f"0{self.qubits}b")
            x, bar_height = left + (index + 0.5) * step, value * chart_height
            y = top + chart_height - bar_height
            body += f'<rect data-basis="{basis}" data-probability="{value!r}" x="{x - step * 0.3:g}" y="{y:g}" width="{step * 0.6:g}" height="{bar_height:g}" rx="3" fill="#2563eb"/>'
            body += text(x, y - 9, f"{value:.1%}", anchor="middle", size=11)
            body += text(x, top + chart_height + 25, f"|{basis}⟩", anchor="middle", size=12)
        order = " ".join(f"q{wire}" for wire in self.wires)
        body += text(left + chart_width / 2, height - 18, f"计算基测量结果 · {order}", anchor="middle", size=11, fill="#64748b")
        return svg(width, height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())


def _bloch_project(x: float, y: float, z: float) -> tuple[float, float]:
    """正交投影：z 向上，x 向左下，y 向右下。"""
    return 240 + 138 * (y - x) / sqrt(2), 235 + 138 * ((x + y) * sin(pi / 8) / sqrt(2) - z * cos(pi / 8))


def _bloch_circle(normal: int) -> str:
    """将三个坐标平面上的大圆投影为球面网格。"""
    points = []
    for step in range(73):
        angle = 2 * pi * step / 72
        coordinates = [cos(angle), sin(angle)]
        coordinates.insert(normal, 0.0)
        x, y = _bloch_project(*coordinates)
        points.append(f'{x:g},{y:g}')
    return 'M' + 'L'.join(points) + 'Z'


class Bloch:
    """单个比特的 Bloch 球；纠缠态显示所选比特的约化态，球心代表最大混合态。"""
    def __init__(self, state: State, wire: int = 0, *, title: str = "") -> None:
        if not isinstance(wire, int): raise TypeError("Bloch 球需要指定一个整数比特编号")
        selected_wires(state.qubits, wire)
        self.wire, self.title = wire, title
        self.vector = tuple(state.expectation([X(wire), Y(wire), Z(wire)]))
        self.length = min(1.0, sqrt(sum(value * value for value in self.vector)))

    def _repr_svg_(self) -> str:
        width, height = 480, 490
        body = text(width / 2, 30, self.title or f"q{self.wire} · Bloch 球", anchor="middle", size=18)
        body += ('<defs><radialGradient id="bloch-shade" cx="35%" cy="25%" r="75%">'
                 '<stop offset="0" stop-color="#f8fbff"/><stop offset="1" stop-color="#dbeafe"/>'
                 '</radialGradient><marker id="bloch-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                 'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                 '<path d="M0,0L10,5L0,10Z" fill="#dc2626"/></marker></defs>')
        body += '<circle data-role="sphere" cx="240" cy="235" r="138" fill="url(#bloch-shade)" stroke="#93b4da" stroke-width="1.5"/>'
        for normal in range(3):
            body += f'<path d="{_bloch_circle(normal)}" fill="none" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 4" opacity=".7"/>'
        for axis, name in enumerate(('x', 'y', 'z')):
            positive, negative = [0.0] * 3, [0.0] * 3
            positive[axis], negative[axis] = 1.16, -1.09
            px, py = _bloch_project(*positive)
            nx, ny = _bloch_project(*negative)
            body += f'<path d="M{nx:g},{ny:g}L240,235" stroke="#94a3b8" stroke-dasharray="3 4"/>'
            body += f'<path data-axis="{name}" d="M240,235L{px:g},{py:g}" stroke="#64748b" stroke-width="1.4"/>'
            body += text(px, py - 9 if name == 'z' else py + 19, name, anchor="middle", size=14)
        for z, label in ((1, '|0⟩'), (-1, '|1⟩')):
            x, y = _bloch_project(0, 0, z)
            body += f'<circle cx="{x:g}" cy="{y:g}" r="3" fill="#64748b"/>'
            body += text(x + 14, y + 5, label, size=13)
        x, y = _bloch_project(*self.vector)
        body += '<circle cx="240" cy="235" r="3" fill="#64748b"/>'
        if self.length > 1e-12:
            body += f'<path data-role="bloch-vector" d="M240,235L{x:g},{y:g}" stroke="#dc2626" stroke-width="3" marker-end="url(#bloch-arrow)"/>'
        body += f'<circle data-role="bloch-point" data-vector="{",".join(map(str, self.vector))}" cx="{x:g}" cy="{y:g}" r="5" fill="#dc2626" stroke="white" stroke-width="1.5"/>'
        coordinates = ', '.join(f'{0.0 if abs(value) < 1e-12 else value:.3f}' for value in self.vector)
        kind = "纯态" if abs(self.length - 1) < 1e-10 else "最大混合态" if self.length < 1e-12 else "混合态"
        body += text(width / 2, 420, f'q{self.wire} · (x, y, z) = ({coordinates})', anchor="middle", size=13)
        body += text(width / 2, 445, f'|r| = {self.length:.3f} · {kind}', anchor="middle", size=13, fill="#dc2626")
        body += text(width / 2, 474, '球面是纯态；球内是混合态。纠缠态显示单个比特的约化态。', anchor="middle", size=11, fill="#64748b")
        return svg(width, height, self.title or f"q{self.wire} · Bloch 球", body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())
