from collections import deque
from collections.abc import Iterable
from html import escape
from math import cos, pi, sin
from pathlib import Path

from ._svg import save_svg, svg, text


type Node = int | str


class Graph:
    """环形布局的无向图；groups 为节点分组，跨组边以绿色显示。"""
    def __init__(self, nodes: Iterable[Node], edges: Iterable[tuple[Node, Node]], *, groups: dict[Node, int] | None = None, title: str = "") -> None:
        self.nodes = tuple(dict.fromkeys(nodes))
        self.edges = tuple(tuple(edge) for edge in edges)
        self.groups, self.title = dict(groups or {}), title
        if any(source not in self.nodes or target not in self.nodes for source, target in self.edges):
            raise ValueError("无向图的连线必须连接已有节点")
        if any(node not in self.nodes for node in self.groups): raise ValueError("分组只能包含已有节点")

    def _repr_svg_(self) -> str:
        width, height = 480, 430
        positions = {node: (240 + 130 * cos(-3 * pi / 4 + 2 * pi * index / len(self.nodes)),
                            210 + 130 * sin(-3 * pi / 4 + 2 * pi * index / len(self.nodes)))
                     for index, node in enumerate(self.nodes)}
        body = text(width / 2, 28, self.title, anchor="middle", size=18)
        cut = 0
        for source, target in self.edges:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            crossing = source in self.groups and target in self.groups and self.groups[source] != self.groups[target]
            cut += crossing
            color = "#16a34a" if crossing else "#94a3b8"
            body += (f'<line data-source="{escape(str(source), quote=True)}" data-target="{escape(str(target), quote=True)}" '
                     f'data-cut="{str(crossing).lower()}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
                     f'stroke="{color}" stroke-width="{4 if crossing else 2}"/>')
        colors = ("#dbeafe", "#ffedd5", "#ede9fe", "#dcfce7")
        for node, (x, y) in positions.items():
            fill = colors[self.groups[node] % len(colors)] if node in self.groups else "#f1f5f9"
            body += f'<g data-node="{escape(str(node), quote=True)}"><circle cx="{x:g}" cy="{y:g}" r="21" fill="{fill}" stroke="#64748b" stroke-width="1.5"/>'
            body += text(x, y + 5, node, anchor="middle", size=14) + '</g>'
        if not self.nodes: body += text(width / 2, 210, "暂无节点", anchor="middle")
        if self.groups:
            body += text(width / 2, 383, f"跨组边：{cut} / {len(self.edges)}", anchor="middle", size=14)
            body += text(width / 2, 412, "不同颜色表示分组；绿色边连接不同分组。", anchor="middle", size=11, fill="#64748b")
        else:
            body += text(width / 2, 402, "无向图；每条边连接两个节点。", anchor="middle", size=11, fill="#64748b")
        return svg(width, height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())


def _levels(nodes, edges):
    incoming = dict.fromkeys(nodes, 0)
    following = {node: [] for node in nodes}
    levels = dict.fromkeys(nodes, 0)
    for edge in edges:
        source, target = edge[:2]
        if source not in nodes or target not in nodes: raise ValueError("结构图的连线必须连接已有节点")
        incoming[target] += 1
        following[source].append(target)
    ready = deque(node for node in nodes if incoming[node] == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in following[node]:
            levels[target] = max(levels[target], levels[node] + 1)
            incoming[target] -= 1
            if incoming[target] == 0: ready.append(target)
    if visited != len(nodes): raise ValueError("结构图暂时只支持没有循环的有向图")
    return levels


class Diagram:
    """nodes 为 {编号: 显示文本} 或字符串列表，edges 为二元或带标签的三元组。"""
    def __init__(self, nodes: dict[str, str] | Iterable[str], edges: Iterable[tuple[str, str] | tuple[str, str, str]], title: str = "") -> None:
        self.nodes = dict(nodes) if isinstance(nodes, dict) else {node: node for node in nodes}
        self.edges = [tuple(edge) for edge in edges]
        self.title = title

    def _repr_svg_(self) -> str:
        levels = _levels(self.nodes, self.edges)
        groups = {}
        for node, level in levels.items(): groups.setdefault(level, []).append(node)
        width = max(360, max((len(group) for group in groups.values()), default=1) * 220 + 40)
        height = max(180, (max(levels.values(), default=0) + 1) * 110 + 55)
        positions = {node: (width / 2 + (column - (len(group) - 1) / 2) * 220, 85 + level * 110)
                     for level, group in groups.items() for column, node in enumerate(group)}
        body = text(width / 2, 28, self.title, anchor="middle", size=18)
        body += '<defs><marker id="qutai-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#64748b"/></marker></defs>'
        for edge in self.edges:
            x1, y1 = positions[edge[0]]
            x2, y2 = positions[edge[1]]
            start, end = y1 + 30, y2 - 32
            middle = (start + end) / 2
            label_x, anchor = (x1 + x2) / 2 + 7, "start"
            if levels[edge[1]] - levels[edge[0]] > 1:
                # 跨越中间层的连线从外侧绕行，避免穿过其他节点。
                route = width - 12 if x1 >= width / 2 else 12
                target = x2 + 97 if route > width / 2 else x2 - 97
                path = f'M{x1:g},{start:g}C{x1:g},{start + 18:g} {route:g},{start + 18:g} {route:g},{start + 40:g}V{y2:g}H{target:g}'
                label_x, anchor = (route - 8, "end") if route > width / 2 else (route + 8, "start")
            else:
                path = f'M{x1:g},{start:g}C{x1:g},{middle:g} {x2:g},{middle:g} {x2:g},{end:g}'
            body += f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#qutai-arrow)"/>'
            if len(edge) > 2: body += text(label_x, middle - 5, edge[2], anchor=anchor, size=10)
        for node, label in self.nodes.items():
            x, y = positions[node]
            fill = "#eef2ff" if str(label).startswith(("ResNetBlock", "TransformerBlock", "DiTBlock", "PQC")) else "#f1f5f9"
            body += f'<g data-node="{escape(str(node), quote=True)}"><rect x="{x - 95:g}" y="{y - 30:g}" width="190" height="60" rx="9" fill="{fill}" stroke="#94a3b8"/>'
            lines = str(label).splitlines() or [""]
            for index, line in enumerate(lines):
                body += text(x, y + (index - (len(lines) - 1) / 2) * 17 + 4, line, anchor="middle", size=12)
            body += '</g>'
        if not self.nodes: body += text(width / 2, 100, "暂无节点", anchor="middle")
        return svg(width, height, self.title, body)

    def save(self, path: str | Path) -> Path:
        return save_svg(path, self._repr_svg_())


def _node(nodes, label):
    name = f"n{len(nodes)}"
    nodes[name] = label
    return name


def _label(layer):
    kind = type(layer).__name__
    weight = getattr(layer, "weight", None)
    shape = getattr(weight, "shape", ())
    if kind == "PQC": return f"PQC\n{layer.qubits} qubits · depth {layer.depth}\n{layer.entangler.upper()}"
    if kind == "Embedding" and len(shape) == 2: return f"Embedding\n{shape[0]} ids → {shape[1]} features"
    if kind == "Linear" and len(shape) == 2: return f"Linear\n{shape[0]} → {shape[1]}"
    if kind == "Conv2d" and len(shape) == 4: return f"Conv2d\n{shape[1]} → {shape[0]} · {shape[2]}×{shape[3]}"
    if kind == "BatchNorm2d" and shape: return f"BatchNorm2d\n{shape[0]} channels"
    if kind == "LayerNorm" and shape: return f"LayerNorm\n{shape[0]} features"
    if kind == "Mean": return f"Mean\naxis={layer.axis}"
    if kind == "TimeEmbedding": return f"TimeEmbedding\n{layer.features} features"
    if kind == "PatchEmbedding":
        shape = layer.projection.weight.shape
        height, width = layer.patch_size
        return f"PatchEmbedding\n{shape[1]} → {shape[0]} · {height}×{width}"
    if kind == "PositionEmbedding" and shape:
        return f"PositionEmbedding\n{shape[0]} tokens · {shape[1]} features"
    if kind == "Attention":
        shape = layer.query.weight.shape
        return f"Attention\n{shape[0]} → {shape[1]}"
    if kind == "MultiHeadAttention":
        shape = layer.query.weight.shape
        return f"MultiHeadAttention\n{shape[0]} features · {layer.heads} heads"
    if kind == "TransformerBlock":
        shape = layer.attention.query.weight.shape
        return f"TransformerBlock\n{shape[0]} features · {layer.attention.heads} heads"
    if kind == "DiTBlock":
        shape = layer.attention.query.weight.shape
        return f"DiTBlock\n{shape[0]} features · {layer.attention.heads} heads"
    if kind == "ResNetBlock":
        main = layer.layers[0].branch.layers[0]
        shape = main.weight.shape
        return f"ResNetBlock\n{shape[1]} → {shape[0]} · stride {main.stride[0]}"
    return kind


def _network_layer(layer, incoming, nodes, edges):
    kind = type(layer).__name__
    if kind in ("ResNetBlock", "TransformerBlock", "DiTBlock", "TimeEmbedding"):
        result = _node(nodes, _label(layer))
        edges.append((incoming, result))
        return result
    if hasattr(layer, "branch") and hasattr(layer, "shortcut"):
        main = _network_layer(layer.branch, incoming, nodes, edges)
        if main == incoming:
            main = _node(nodes, "Identity")
            edges.append((incoming, main))
        if layer.shortcut is None:
            shortcut = _node(nodes, "Identity")
            edges.append((incoming, shortcut))
        else:
            shortcut = _network_layer(layer.shortcut, incoming, nodes, edges)
            if shortcut == incoming:
                shortcut = _node(nodes, "Identity")
                edges.append((incoming, shortcut))
        result = _node(nodes, "Add (+)")
        edges.extend(((main, result, "main"), (shortcut, result, "shortcut")))
        return result
    if hasattr(layer, "layers"):
        for child in layer.layers: incoming = _network_layer(child, incoming, nodes, edges)
        return incoming
    result = _node(nodes, _label(layer))
    edges.append((incoming, result))
    return result


def _dit_network(model):
    """图片与时间分别编码，在各个 DiTBlock 汇合，不执行前向计算。"""
    nodes, edges = {}, []
    channels, height, width = model.image_shape
    image = _node(nodes, f"Image\n{channels}×{height}×{width}")
    time = _node(nodes, "Time t")
    tokens = _network_layer(model.patch_embedding, image, nodes, edges)
    tokens = _network_layer(model.position_embedding, tokens, nodes, edges)
    condition = _network_layer(model.time_embedding, time, nodes, edges)
    for block in model.blocks:
        tokens = _network_layer(block, tokens, nodes, edges)
        edges.append((condition, tokens, "time"))
    tokens = _network_layer(model.output_norm, tokens, nodes, edges)
    patches = _network_layer(model.output, tokens, nodes, edges)
    image = _node(nodes, f"Unpatchify\n{channels}×{height}×{width}")
    edges.append((patches, image))
    edges.append((image, _node(nodes, "Output image")))
    return nodes, edges


class Network(Diagram):
    """只读取结构；组合块折叠显示，DiT 保留图片与时间两路输入。"""
    def __init__(self, model: object, title: str = "") -> None:
        network = getattr(model, "network", model)
        if type(network).__name__ == "DiT":
            nodes, edges = _dit_network(network)
            super().__init__(nodes, edges, title=title)
            return
        nodes, edges = {}, []
        tokenizer = getattr(model, "tokenizer", None)
        context = getattr(model, "context_length", None)
        language_model = tokenizer is not None and context is not None
        incoming = _node(nodes, f"Character ids\ncontext ≤ {context}" if language_model else "Input")
        output = _network_layer(model, incoming, nodes, edges)
        label = f"Next character\n{tokenizer.size} scores/token" if language_model else "Output"
        edges.append((output, _node(nodes, label)))
        super().__init__(nodes, edges, title=title)
