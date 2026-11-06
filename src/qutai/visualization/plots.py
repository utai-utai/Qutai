from collections.abc import Iterable
from pathlib import Path
from typing import Self

from ._svg import save_svg, svg, text


_COLORS = ("#2563eb", "#dc6b19", "#0f956f", "#9b51c6", "#d64064", "#64748b")


def _limits(values: list[float]) -> tuple[float, float]:
    if not values: return 0.0, 1.0
    low, high = min(values), max(values)
    margin = (high - low) * 0.05 if low != high else max(abs(low) * 0.05, 1.0)
    return low - margin, high + margin


class Plot:
    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.series = []

    def _add(self, kind: str, x: Iterable[float], y: Iterable[float], label: str) -> Self:
        x, y = list(map(float, x)), list(map(float, y))
        if len(x) != len(y): raise ValueError("横坐标和纵坐标的数量必须相同")
        self.series.append((kind, x, y, label))
        return self

    def line(self, x: Iterable[float], y: Iterable[float], label: str = "") -> Self:
        return self._add("line", x, y, label)

    def scatter(self, x: Iterable[float], y: Iterable[float], label: str = "") -> Self:
        return self._add("scatter", x, y, label)

    def _repr_svg_(self) -> str:
        left, top, width, height = 75, 55, 610, 260
        legends = [(index, series[3]) for index, series in enumerate(self.series) if series[3]]
        total_height = 385 + ((len(legends) + 2) // 3) * 24
        xmin, xmax = _limits([value for _, x, _, _ in self.series for value in x])
        ymin, ymax = _limits([value for _, _, y, _ in self.series for value in y])
        body = text(360, 28, self.title, anchor="middle", size=18)
        for tick in range(6):
            x, y = left + width * tick / 5, top + height - height * tick / 5
            body += f'<path d="M{x:g},{top}V{top + height} M{left},{y:g}H{left + width}" stroke="#e2e8f0" fill="none"/>'
            body += text(x, top + height + 24, f"{xmin + (xmax - xmin) * tick / 5:.3g}", anchor="middle")
            body += text(left - 10, y + 4, f"{ymin + (ymax - ymin) * tick / 5:.3g}", anchor="end")
        body += f'<path d="M{left},{top}V{top + height}H{left + width}" stroke="#64748b" fill="none"/>'
        body += text(left + width / 2, top + height + 52, self.xlabel, anchor="middle")
        body += f'<g transform="rotate(-90 18 {top + height / 2})">{text(18, top + height / 2, self.ylabel, anchor="middle")}</g>'
        for index, (kind, xvalues, yvalues, _) in enumerate(self.series):
            color = _COLORS[index % len(_COLORS)]
            points = [(left + (x - xmin) * width / (xmax - xmin), top + height - (y - ymin) * height / (ymax - ymin))
                      for x, y in zip(xvalues, yvalues, strict=True)]
            if kind == "line" and points:
                coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                body += f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            if kind == "scatter" or len(points) == 1:
                body += ''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>' for x, y in points)
        for position, (index, label) in enumerate(legends):
            x, y = left + position % 3 * 205, 390 + position // 3 * 24
            body += f'<path d="M{x},{y - 4}h18" stroke="{_COLORS[index % len(_COLORS)]}" stroke-width="3"/>'
            body += text(x + 25, y, label)
        return svg(720, total_height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())


class Heatmap:
    def __init__(self, matrix: Iterable[Iterable[float]], xlabels: Iterable[object] | None = None, ylabels: Iterable[object] | None = None, title: str = "") -> None:
        self.matrix = [list(map(float, row)) for row in matrix]
        columns = len(self.matrix[0]) if self.matrix else 0
        if any(len(row) != columns for row in self.matrix): raise ValueError("热力图每行的列数必须相同")
        self.xlabels = list(range(columns)) if xlabels is None else list(xlabels)
        self.ylabels = list(range(len(self.matrix))) if ylabels is None else list(ylabels)
        if len(self.xlabels) != columns or len(self.ylabels) != len(self.matrix): raise ValueError("热力图标签数量必须与矩阵形状一致")
        self.title = title

    def _repr_svg_(self) -> str:
        rows, columns = len(self.matrix), len(self.xlabels)
        left = max(65, min(180, max((len(str(label)) * 7 + 15 for label in self.ylabels), default=0)))
        cell, top = 54, 95
        width, height = max(300, left + columns * cell + 25), max(180, top + rows * cell + 60)
        body = text(width / 2, 28, self.title, anchor="middle", size=18)
        values = [value for row in self.matrix for value in row]
        if not values: return svg(width, height, self.title, body + text(width / 2, 115, "暂无数据", anchor="middle"))
        low, high = min(values), max(values)
        for column, label in enumerate(self.xlabels):
            x = left + (column + 0.5) * cell
            body += f'<g transform="rotate(-35 {x:g} {top - 12})">{text(x, top - 12, label, size=11)}</g>'
        for row, label in enumerate(self.ylabels):
            body += text(left - 10, top + (row + 0.5) * cell + 4, label, anchor="end", size=11)
        for row, data in enumerate(self.matrix):
            for column, value in enumerate(data):
                ratio = (value - low) / (high - low) if high != low else 0.5
                color = '#%02x%02x%02x' % tuple(round(a + (b - a) * ratio) for a, b in zip((237, 242, 255), (30, 64, 175), strict=True))
                x, y = left + column * cell, top + row * cell
                body += f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/>'
                body += text(x + cell / 2, y + cell / 2 + 4, f"{value:.3g}", anchor="middle", fill="white" if ratio > 0.55 else "#1e293b")
        body += text(left, height - 18, f"数值范围：{low:.3g} — {high:.3g}", size=11)
        return svg(width, height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())
