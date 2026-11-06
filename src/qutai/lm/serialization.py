import json
from pathlib import Path

from ..nn.model import Model as NeuralModel
from ..nn.serialization import describe_model, read_file
from .tokenizer import Tokenizer


def save_model(network: NeuralModel, tokenizer: Tokenizer, context_length: int, path: str | Path) -> None:
    """一次写入网络、参数、字符编号顺序和生成时使用的上下文长度。"""
    data = describe_model(network, NeuralModel)
    data.update(characters=tokenizer.characters, context_length=context_length)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(path: str | Path) -> tuple[Tokenizer, int]:
    """恢复词表并核对输入表、位置长度和输出类别数，避免字符编号错配。"""
    data = read_file(path)
    if not isinstance(data, dict) or "characters" not in data or "context_length" not in data:
        raise ValueError("文件缺少语言模型的词表或上下文长度")
    tokenizer, length = Tokenizer(data["characters"]), data["context_length"]
    if not tokenizer.size or tokenizer.size != len(data["characters"]) or length <= 0:
        raise ValueError("文件中的词表或上下文长度无效")
    layers = data["model"]["layers"]
    if len(layers) < 3 or (layers[0]["type"], layers[1]["type"], layers[-1]["type"]) != ("Embedding", "PositionEmbedding", "Linear"):
        raise ValueError("语言模型需要字符嵌入、位置嵌入和词表输出层")
    if layers[0]["count"] != tokenizer.size or layers[-1]["out_features"] != tokenizer.size or layers[1]["length"] != length:
        raise ValueError("词表或上下文长度与网络结构不一致")
    return tokenizer, length
