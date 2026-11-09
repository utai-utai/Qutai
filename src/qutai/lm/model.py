from math import exp, isfinite
from pathlib import Path
from random import Random

from ..core.context import set_recording, set_training
from ..core.migration import move
from ..core.tensor import Tensor
from ..core.types import IntegerData, LayerLike
from ..nn.cache import KVCache
from ..nn.layers import Embedding, LayerNorm, Linear, PositionEmbedding, TransformerBlock
from ..nn.model import Model as NeuralModel
from ..nn.serialization import load_into, restore_layers
from .serialization import read_metadata, save_model
from .tokenizer import Tokenizer


class Model:
    def __init__(self, tokenizer: Tokenizer | None = None, *, context_length: int = 16, features: int = 16, heads: int = 2, depth: int = 1, hidden_features: int | None = None, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        if context_length <= 0 or depth <= 0: raise ValueError("上下文长度和 Transformer 层数必须大于零")
        self._device, self._dtype = device, dtype
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.embedding: Embedding | None = None
        self.network = NeuralModel()
        if tokenizer is not None:
            if tokenizer.size == 0: raise ValueError("语言模型的词表不能为空")
            self.embedding = Embedding(tokenizer.size, features, seed=seed)
            self.network = NeuralModel([
                PositionEmbedding(context_length, features, seed=None if seed is None else seed + 1),
                *[TransformerBlock(features, heads, hidden_features=hidden_features, causal=True,
                                   seed=None if seed is None else seed + 2 + index * 6) for index in range(depth)],
                LayerNorm(features),
                Linear(features, tokenizer.size, seed=None if seed is None else seed + 2 + depth * 6),
            ])
        self.to(device, dtype=dtype)

    @property
    def layers(self) -> tuple[LayerLike[Tensor], ...]:
        """按实际计算顺序展示字符嵌入和后面的浮点网络。"""
        return () if self.embedding is None else (self.embedding, *self.network.layers)

    def to(self, device: str | None = None, *, dtype: str | None = None) -> Model:
        """原地迁移模型参数、已有梯度与子层的运行统计量。"""
        move(self, device, dtype)
        if device is not None: self._device, self._dtype = device, dtype
        elif dtype is not None: self._dtype = dtype
        return self

    def parameters(self) -> tuple[Tensor, ...]:
        """参数收集与去重复用神经网络容器，SGD 可以直接接收本模型。"""
        return NeuralModel(self.layers).parameters()

    def __call__(self, tokens: IntegerData, *, train: bool = True) -> Tensor:
        """输入二维整数编号，输出（批次，长度，词表大小）的原始得分。"""
        if self.embedding is None: raise ValueError("请先创建或加载语言模型")
        with set_recording(train), set_training(train):
            tensor = self.embedding(tokens)
            if tensor.ndim != 3 or tensor.shape[1] == 0: raise ValueError("语言模型需要非空的二维字符编号")
            return self.network(tensor, train=train)

    def _cached(self, tokens: IntegerData, cache: dict[int, KVCache], *, offset: int) -> Tensor:
        """只计算本次输入的位置；每个 Transformer 层的位置各自保存 K/V。"""
        if self.embedding is None: raise ValueError("请先创建或加载语言模型")
        with set_recording(False), set_training(False):
            tensor = self.embedding(tokens)
            for index, layer in enumerate(self.network.layers):
                if isinstance(layer, PositionEmbedding): tensor = layer(tensor, offset=offset)
                elif isinstance(layer, TransformerBlock): tensor = layer(tensor, cache=cache[index])
                else: tensor = layer(tensor)
            return tensor

    def generate(self, prompt: str, *, length: int = 20, temperature: float = 0.0, top_k: int | None = None, seed: int | None = None, use_cache: bool = True) -> str:
        """逐字续写，默认复用 K/V；温度为零时取最高分，正温度按概率采样。"""
        if self.tokenizer is None: raise ValueError("请先创建或加载语言模型")
        if not prompt or length < 0: raise ValueError("续写开头不能为空，新增字符数不能为负数")
        if not isfinite(temperature) or temperature < 0: raise ValueError("采样温度必须是有限的非负数")
        if top_k is not None and (type(top_k) is not int or top_k <= 0): raise ValueError("top_k 必须是正整数或 None")
        random = Random(seed)
        tokens = self.tokenizer.encode(prompt)
        cache: dict[int, KVCache] = {}
        offset = 0
        for _ in range(length):
            if use_cache:
                if offset == 0 or offset >= self.context_length:
                    # 移窗会改变位置与前文，重建完整窗口才能保持原来的计算结果。
                    cache = {index: KVCache() for index, layer in enumerate(self.network.layers) if isinstance(layer, TransformerBlock)}
                    current, offset = tokens[-self.context_length:], 0
                else:
                    current = tokens[-1:]
                scores = self._cached([current], cache, offset=offset)[0, -1].tolist()
                offset += len(current)
            else:
                scores = self([tokens[-self.context_length:]], train=False)[0, -1].tolist()
            if temperature == 0:
                next_token = max(range(self.tokenizer.size), key=scores.__getitem__)
            else:
                candidates = sorted(range(self.tokenizer.size), key=scores.__getitem__, reverse=True)[:top_k]
                maximum = scores[candidates[0]]
                weights = [exp((scores[index] - maximum) / temperature) for index in candidates]
                next_token = random.choices(candidates, weights=weights)[0]
            tokens.append(next_token)
        return self.tokenizer.decode(tokens)

    def save(self, path: str | Path) -> None:
        """同时保存网络、参数、词表和上下文长度。"""
        if self.tokenizer is None: raise ValueError("请先创建或加载语言模型")
        save_model(NeuralModel(self.layers), self.tokenizer, self.context_length, path)

    def load(self, path: str | Path) -> Model:
        """空模型恢复全部信息；已有模型核对词表和结构后原地加载，返回自身。"""
        tokenizer, context_length = read_metadata(path)
        if self.tokenizer is not None:
            if self.tokenizer.characters != tokenizer.characters or self.context_length != context_length:
                raise ValueError("文件的词表顺序或上下文长度与当前模型不一致")
            load_into(NeuralModel(self.layers), path, NeuralModel)
        else:
            layers = restore_layers(path, NeuralModel)
            embedding = layers[0]
            if not isinstance(embedding, Embedding): raise ValueError("语言模型的第一层必须是字符嵌入")
            self.embedding = embedding
            self.network = NeuralModel(layers[1:])
            self.tokenizer, self.context_length = tokenizer, context_length
            self.to(self._device, dtype=self._dtype)
        return self
