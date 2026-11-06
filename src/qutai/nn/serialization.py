import json
from collections.abc import Callable, Iterable
from math import prod
from pathlib import Path

from ..backends.device import get_backend, transfer
from ..core.tensor import Tensor
from ..core.types import LayerLike
from ..qnn.pqc import PQC
from .dit import DiT, DiTBlock, TimeEmbedding
from .layers import (
    Attention, BatchNorm2d, Conv2d, Embedding, FeedForward, Flatten, GELU, GlobalAvgPool2d,
    Identity, LayerNorm, Linear, MaxPool2d, Mean, MultiHeadAttention,
    PatchEmbedding, PositionEmbedding, ReLU, Residual, ResNetBlock, Sequence,
    Softmax, TransformerBlock,
)


# 组合层需要递归保存的子层，共用同一份关系完成描述、恢复和状态复制。
_CHILDREN = {
    "Attention": ("query", "key", "value"),
    "MultiHeadAttention": ("query", "key", "value", "output"),
    "TransformerBlock": ("attention_norm", "attention", "feed_forward_norm", "feed_forward"),
    "PatchEmbedding": ("projection",),
    "DiTBlock": ("attention_norm", "attention", "feed_forward_norm", "feed_forward", "modulation"),
    "DiT": ("patch_embedding", "position_embedding", "time_embedding", "output_norm", "output"),
}


def describe_model(model: LayerLike[Tensor], model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]]) -> dict:
    """生成结构和参数表，供神经网络与语言模型共用同一份保存逻辑。"""
    parameters = model.parameters()
    indices = {parameter: index for index, parameter in enumerate(parameters)}
    structure = _describe_layer(model, indices, model_type)
    records = [{"shape": parameter.shape, "values": parameter.storage.tolist(), "requires_grad": parameter.requires_grad, "dtype": parameter.dtype} for parameter in parameters]
    return {"model": structure, "parameters": records}


def save_model(model: LayerLike[Tensor], path: str | Path, model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]]) -> None:
    """把模型结构、参数数值和可训练标记写入 JSON 文件。"""
    Path(path).write_text(json.dumps(describe_model(model, model_type), indent=2), encoding="utf-8")


def _describe_layer(layer: LayerLike[Tensor], indices: dict[Tensor, int], model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]], normalizations: dict[BatchNorm2d, int] | None = None) -> dict:
    """将一层描述成字典，递归保留共享参数与批归一化状态的关系。"""
    if normalizations is None: normalizations = {}
    if type(layer) in (model_type, Sequence, ResNetBlock, FeedForward, TimeEmbedding):
        structure: dict = {"type": "Model", "layers": [_describe_layer(child, indices, model_type, normalizations) for child in getattr(layer, "layers")]}
        if type(layer) is ResNetBlock:
            structure.update(type="ResNetBlock", in_channels=layer.in_channels, out_channels=layer.out_channels, stride=layer.stride)
        elif type(layer) is FeedForward:
            structure.update(type="FeedForward", features=layer.layers[0].weight.shape[0], hidden_features=layer.layers[0].weight.shape[1])
        elif type(layer) is TimeEmbedding:
            structure.update(type="TimeEmbedding", features=layer.features)
        return structure
    if type(layer) is Identity: return {"type": "Identity"}
    if type(layer) is Linear: return {"type": "Linear", "in_features": layer.weight.shape[0], "out_features": layer.weight.shape[1], "parameters": [indices[layer.weight], indices[layer.bias]]}
    if type(layer) is PQC:
        return {"type": "PQC", "qubits": layer.qubits, "depth": layer.depth, "entangler": layer.entangler,
                "rotations": [list(row) for row in layer.rotations], "parameters": [indices[layer.weight]]}
    if type(layer) in (Attention, MultiHeadAttention, TransformerBlock, PatchEmbedding, DiTBlock, DiT):
        kind = type(layer).__name__
        structure = {"type": kind, **{name: _describe_layer(getattr(layer, name), indices, model_type, normalizations) for name in _CHILDREN[kind]}}
        if type(layer) is MultiHeadAttention: structure.update(heads=layer.heads, causal=layer.causal)
        elif type(layer) is PatchEmbedding: structure.update(patch_size=list(layer.patch_size))
        elif type(layer) is DiT:
            structure.update(image_shape=list(layer.image_shape), patch_size=list(layer.patch_size),
                             blocks=[_describe_layer(block, indices, model_type, normalizations) for block in layer.blocks])
        return structure
    if type(layer) is LayerNorm:
        return {"type": "LayerNorm", "features": layer.weight.size, "eps": layer.eps, "parameters": [indices[layer.weight], indices[layer.bias]]}
    if type(layer) is PositionEmbedding:
        return {"type": "PositionEmbedding", "length": layer.weight.shape[0], "features": layer.weight.shape[1], "parameters": [indices[layer.weight]]}
    if type(layer) is Embedding:
        return {"type": "Embedding", "count": layer.weight.shape[0], "features": layer.weight.shape[1], "parameters": [indices[layer.weight]]}
    if type(layer) is Conv2d:
        return {"type": "Conv2d", "in_channels": layer.weight.shape[1], "out_channels": layer.weight.shape[0],
                "kernel_size": list(layer.weight.shape[2:]), "stride": list(layer.stride), "padding": list(layer.padding),
                "bias": layer.bias is not None, "parameters": [indices[parameter] for parameter in layer.parameters()]}
    if type(layer) is MaxPool2d: return {"type": "MaxPool2d", "kernel_size": list(layer.kernel_size), "stride": list(layer.stride), "padding": list(layer.padding)}
    if type(layer) is Flatten: return {"type": "Flatten", "start_dim": layer.start_dim}
    if type(layer) is GlobalAvgPool2d: return {"type": "GlobalAvgPool2d"}
    if type(layer) is BatchNorm2d:
        return {"type": "BatchNorm2d", "num_features": layer.weight.size, "eps": layer.eps, "momentum": layer.momentum,
                "state": normalizations.setdefault(layer, len(normalizations)),
                "parameters": [indices[layer.weight], indices[layer.bias]],
                "running_mean": layer.running_mean.tolist(), "running_var": layer.running_var.tolist()}
    if type(layer) is Residual:
        return {"type": "Residual", "branch": _describe_layer(layer.branch, indices, model_type, normalizations),
                "shortcut": None if layer.shortcut is None else _describe_layer(layer.shortcut, indices, model_type, normalizations)}
    if type(layer) is ReLU: return {"type": "ReLU"}
    if type(layer) is GELU: return {"type": "GELU"}
    if type(layer) is Mean: return {"type": "Mean", "axis": layer.axis, "keepdims": layer.keepdims}
    if type(layer) is Softmax: return {"type": "Softmax", "axis": layer.axis}
    raise TypeError(f"暂不支持保存层类型：{type(layer).__name__}")


def _restore_layer(structure: dict, parameters: tuple[Tensor, ...], model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]], normalizations: dict[int, BatchNorm2d] | None = None) -> LayerLike[Tensor]:
    """按编号绑定参数；相同状态编号复用同一个批归一化层。"""
    if normalizations is None: normalizations = {}
    kind = structure["type"]
    if kind == "Model": return model_type(_restore_layer(child, parameters, model_type, normalizations) for child in structure["layers"])
    if kind in ("ResNetBlock", "FeedForward", "TimeEmbedding"):
        if kind == "ResNetBlock":
            block = ResNetBlock(structure["in_channels"], structure["out_channels"], stride=structure["stride"], seed=0)
        elif kind == "FeedForward":
            block = FeedForward(structure["features"], hidden_features=structure["hidden_features"], seed=0)
        else:
            block = TimeEmbedding(structure["features"], seed=0)
        block.layers = tuple(_restore_layer(child, parameters, model_type, normalizations) for child in structure["layers"])
        return block
    if kind == "Identity": return Identity()
    if kind in _CHILDREN:
        if kind == "Attention":
            query = structure["query"]
            layer = Attention(query["in_features"], query["out_features"], seed=0)
        elif kind == "MultiHeadAttention":
            layer = MultiHeadAttention(structure["query"]["in_features"], structure["heads"], causal=structure["causal"], seed=0)
        elif kind == "TransformerBlock":
            attention, feed_forward, norm = structure["attention"], structure["feed_forward"], structure["attention_norm"]
            layer = TransformerBlock(norm["features"], attention["heads"], hidden_features=feed_forward["hidden_features"],
                                     causal=attention["causal"], eps=norm["eps"], seed=0)
        elif kind == "DiTBlock":
            layer = DiTBlock(structure["attention_norm"]["features"], structure["attention"]["heads"],
                             hidden_features=structure["feed_forward"]["hidden_features"], seed=0)
        elif kind == "DiT":
            image_shape = structure["image_shape"]
            layer = DiT(image_shape[0], tuple(image_shape[1:]), patch_size=tuple(structure["patch_size"]),
                        features=structure["patch_embedding"]["projection"]["out_channels"], heads=1, depth=len(structure["blocks"]), seed=0)
            setattr(layer, "blocks", tuple(_restore_layer(block, parameters, model_type, normalizations) for block in structure["blocks"]))
        else:
            projection = structure["projection"]
            layer = PatchEmbedding(projection["in_channels"], projection["out_channels"], tuple(structure["patch_size"]), seed=0)
        for name in _CHILDREN[kind]:
            setattr(layer, name, _restore_layer(structure[name], parameters, model_type, normalizations))
        return layer
    if kind == "ReLU": return ReLU()
    if kind == "GELU": return GELU()
    if kind == "Mean": return Mean(axis=structure["axis"], keepdims=structure["keepdims"])
    if kind == "Softmax": return Softmax(structure["axis"])
    if kind == "MaxPool2d": return MaxPool2d(tuple(structure["kernel_size"]), stride=tuple(structure["stride"]), padding=tuple(structure["padding"]))
    if kind == "Flatten": return Flatten(structure["start_dim"])
    if kind == "GlobalAvgPool2d": return GlobalAvgPool2d()
    if kind == "Residual":
        shortcut = structure["shortcut"]
        return Residual(_restore_layer(structure["branch"], parameters, model_type, normalizations),
                        None if shortcut is None else _restore_layer(shortcut, parameters, model_type, normalizations))
    if kind in ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm", "PositionEmbedding", "Embedding", "PQC"):
        if kind == "PQC":
            layer = PQC(structure["qubits"], structure["depth"], entangler=structure["entangler"], seed=0)
            rotations = tuple(tuple(row) for row in structure["rotations"])
            if (len(rotations) != layer.depth or any(len(row) != layer.qubits for row in rotations)
                    or any(axis not in ("RX", "RY", "RZ") for row in rotations for axis in row)):
                raise ValueError("PQC 的旋转门结构与比特数或深度不一致")
            layer.rotations = rotations
        elif kind == "Linear":
            layer = Linear(structure["in_features"], structure["out_features"], seed=0)
        elif kind == "Conv2d":
            layer = Conv2d(structure["in_channels"], structure["out_channels"], tuple(structure["kernel_size"]), stride=tuple(structure["stride"]), padding=tuple(structure["padding"]), bias=structure["bias"], seed=0)
        elif kind == "LayerNorm":
            layer = LayerNorm(structure["features"], eps=structure["eps"])
        elif kind == "PositionEmbedding":
            layer = PositionEmbedding(structure["length"], structure["features"], seed=0)
        elif kind == "Embedding":
            layer = Embedding(structure["count"], structure["features"], seed=0)
        else:
            layer = BatchNorm2d(structure["num_features"], eps=structure["eps"], momentum=structure["momentum"])
            mean, variance = list(map(float, structure["running_mean"])), list(map(float, structure["running_var"]))
            if len(mean) != layer.weight.size or len(variance) != layer.weight.size or any(value < 0 for value in variance): raise ValueError("批归一化的运行统计量与通道配置不一致")
            backend = get_backend(layer.weight.device)
            backend.copy_into(layer.running_mean, backend.storage(mean))
            backend.copy_into(layer.running_var, backend.storage(variance))
        indices = structure["parameters"]
        if any(index < 0 or index >= len(parameters) for index in indices): raise ValueError("文件中的参数索引无效")
        loaded = tuple(parameters[index] for index in indices)
        expected = layer.parameters()
        if len(loaded) != len(expected) or any(value.shape != original.shape for value, original in zip(loaded, expected, strict=True)): raise ValueError("层配置与参数形状不一致")
        layer.weight = loaded[0]
        if len(loaded) == 2: layer.bias = loaded[1]
        if isinstance(layer, BatchNorm2d):
            layer.running_mean = transfer(layer.running_mean, layer.weight.device, layer.weight.dtype)
            layer.running_var = transfer(layer.running_var, layer.weight.device, layer.weight.dtype)
            previous = normalizations.setdefault(structure["state"], layer)
            if (previous.parameters() != layer.parameters() or previous.eps != layer.eps or previous.momentum != layer.momentum
                    or previous.running_mean != layer.running_mean or previous.running_var != layer.running_var):
                raise ValueError("同一批归一化状态的描述不一致")
            return previous
        return layer
    raise ValueError(f"不支持的层类型：{kind}")


def restore_layers(path: str | Path, model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]]) -> tuple[LayerLike[Tensor], ...]:
    """从完整模型文件恢复各层，检查参数引用后交给空模型。"""
    structure, parameters = read_model(path)
    if structure["type"] != "Model": raise ValueError("文件的根结构必须是 Model")
    normalizations: dict[int, BatchNorm2d] = {}
    layers = tuple(_restore_layer(child, parameters, model_type, normalizations) for child in structure["layers"])
    referenced = tuple(dict.fromkeys(parameter for layer in layers for parameter in layer.parameters()))
    if referenced != parameters: raise ValueError("模型参数引用与文件参数表不一致")
    return layers


def load_into(model: LayerLike[Tensor], path: str | Path, model_type: Callable[[Iterable[LayerLike[Tensor]]], LayerLike[Tensor]]) -> None:
    """核对已有模型的结构，再更新参数数值，保留参数对象和优化器引用。"""
    data = read_file(path)
    parameters = model.parameters()
    if isinstance(data, dict):
        indices = {parameter: index for index, parameter in enumerate(parameters)}
        if _configuration(data["model"]) != _configuration(_describe_layer(model, indices, model_type)): raise ValueError("文件中的模型结构与当前模型不一致")
        restored = _restore_layer(data["model"], parameters, model_type)
        load_parameters(parameters, data["parameters"])
        _copy_statistics(model, restored)
    else:
        load_parameters(parameters, data)


def _configuration(structure: dict) -> dict:
    """比较模型配置时排除会随训练改变的运行统计量。"""
    result = {key: value for key, value in structure.items() if key not in ("running_mean", "running_var")}
    if structure["type"] in ("Model", "ResNetBlock", "FeedForward", "TimeEmbedding"):
        result["layers"] = [_configuration(layer) for layer in structure["layers"]]
    elif structure["type"] in _CHILDREN:
        for name in _CHILDREN[structure["type"]]: result[name] = _configuration(structure[name])
        if structure["type"] == "DiT": result["blocks"] = [_configuration(block) for block in structure["blocks"]]
    elif structure["type"] == "Residual":
        result["branch"] = _configuration(structure["branch"])
        result["shortcut"] = None if structure["shortcut"] is None else _configuration(structure["shortcut"])
    return result


def _copy_statistics(target: LayerLike[Tensor], source: LayerLike[Tensor]) -> None:
    """参数检查通过后，原地恢复各批归一化层的运行统计量。"""
    if isinstance(target, BatchNorm2d) and isinstance(source, BatchNorm2d):
        backend = get_backend(target.weight.device)
        backend.copy_into(target.running_mean, source.running_mean)
        backend.copy_into(target.running_var, source.running_var)
    elif isinstance(target, Sequence) and isinstance(source, Sequence):
        for left, right in zip(target.layers, source.layers, strict=True):
            _copy_statistics(left, right)
    elif isinstance(target, Residual) and isinstance(source, Residual):
        _copy_statistics(target.branch, source.branch)
        if target.shortcut is not None and source.shortcut is not None: _copy_statistics(target.shortcut, source.shortcut)
    elif type(target).__name__ in _CHILDREN:
        for name in _CHILDREN[type(target).__name__]: _copy_statistics(getattr(target, name), getattr(source, name))
        if isinstance(target, DiT) and isinstance(source, DiT):
            for left, right in zip(target.blocks, source.blocks, strict=True): _copy_statistics(left, right)


def read_file(path: str | Path) -> dict | list:
    """读取 JSON，返回模型字典或旧版参数列表。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decode_parameters(records: list[dict]) -> tuple[Tensor, ...]:
    """将文件中的参数记录转成 Tensor，并检查形状和元素数量。"""
    parameters = []
    for record in records:
        shape = tuple(record["shape"])
        values = [float(value) for value in record["values"]]
        if any(not isinstance(size, int) or size < 0 for size in shape) or len(values) != prod(shape): raise ValueError("文件中的参数形状与元素数量不一致")
        parameters.append(Tensor(values, shape=shape, requires_grad=record.get("requires_grad", True), dtype=record.get("dtype", "float64")))
    return tuple(parameters)


def read_model(path: str | Path) -> tuple[dict, tuple[Tensor, ...]]:
    """拆出模型结构和 Tensor 参数表；完整重建需要文件包含结构。"""
    data = read_file(path)
    if isinstance(data, list): raise ValueError("文件只有参数，缺少模型结构，无法自动加载")
    return data["model"], _decode_parameters(data["parameters"])


def load_parameters(parameters: Iterable[Tensor], records: list[dict]) -> None:
    """检查全部参数后填回数值，保留参数对象并清除旧梯度。"""
    parameters = tuple(parameters)
    if len(records) != len(parameters): raise ValueError("文件中的参数数量与当前模型不一致")
    loaded = _decode_parameters(records)
    if any(value.shape != parameter.shape for parameter, value in zip(parameters, loaded, strict=True)): raise ValueError("文件中的参数形状与当前模型不一致")
    for parameter, value in zip(parameters, loaded, strict=True):
        get_backend(parameter.device).copy_into(parameter.storage, transfer(value.storage, parameter.device, parameter.dtype))
        parameter.zero_grad()
