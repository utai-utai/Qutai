from ..core.context import set_recording, set_training
from ..backends.device import default_dtype, get_backend
from ..core.migration import move
from ..core.shape import flatten_indices, infer_shape
from ..core.tensor import Tensor
from .config import Config
from .layers import (
    Block,
    Projection,
    RMSNorm,
    backend,
    embed,
    linear,
    projection_loss,
    unpack,
)
from .recompute import checkpoint


class Decoder:
    def __init__(self, config: Config, *, device="cpu", dtype=None, seed=None):
        dtype = default_dtype(device) if dtype is None else dtype
        if dtype not in ("float32", "float64"):
            raise ValueError("解码器参数必须为 float32 或 float64")
        self.config = config
        kw = dict(device=device, dtype=dtype)
        self.embedding = Projection(config.d_model, config.vocab_size, seed=seed, **kw)
        self.blocks = tuple(
            (
                Block(config, seed=None if seed is None else seed + 1 + 100 * i, **kw)
                for i in range(config.n_layers)
            )
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps, **kw)
        self.last_lm_loss = None
        self.last_auxiliary_loss = None

    def named_parameters(self):
        result = [("embedding.weight", self.embedding.weight)]
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            result.extend(
                (
                    (prefix + name + ".weight", getattr(block, name).weight)
                    for name in ("attn_norm", "ffn_norm")
                )
            )
            result.extend(
                (
                    (
                        prefix + "attn." + name + ".weight",
                        getattr(block.attn, name).weight,
                    )
                    for name in ("q", "k", "v", "out")
                )
            )
            if block.is_moe:
                result.append((prefix + "ffn.router.weight", block.ffn.router.weight))
                for e, expert in enumerate(block.ffn.experts):
                    result.extend(
                        (
                            (
                                prefix + f"ffn.experts.{e}." + name + ".weight",
                                getattr(expert, name).weight,
                            )
                            for name in ("gate", "up", "down")
                        )
                    )
            else:
                result.extend(
                    (
                        (
                            prefix + "ffn." + name + ".weight",
                            getattr(block.ffn, name).weight,
                        )
                        for name in ("gate", "up", "down")
                    )
                )
        result.append(("norm.weight", self.norm.weight))
        return tuple(result)

    def parameters(self):
        return tuple(
            dict.fromkeys((parameter for _, parameter in self.named_parameters()))
        )

    def parameter_counts(self):
        return self.config.parameter_counts()

    def to(self, device=None, *, dtype=None):
        return move(self, device, dtype)

    def _tokens(self, tokens):
        shape = infer_shape(tokens)
        if (
            len(shape) != 2
            or not shape[0]
            or (not 1 <= shape[1] <= self.config.max_seq_len)
        ):
            raise ValueError("字符编号需要非空二维批次，且长度不超过模型上下文")
        ids = tuple(flatten_indices(tokens))
        if any((token < 0 or token >= self.config.vocab_size for token in ids)):
            raise ValueError("字符编号超出词表范围")
        return (ids, shape)

    def hidden(self, tokens, *, train=True):
        with set_recording(train), set_training(train):
            return self._hidden(tokens, train=train)

    def _hidden(self, tokens, *, train=True):
        ids, shape = self._tokens(tokens)
        hidden = embed(ids, self.embedding.weight, shape)
        auxiliary = []
        for block in self.blocks:
            if train and self.config.activation_checkpointing:
                if block.is_moe:
                    packed = checkpoint(
                        block.packed, hidden, parameters=block.parameters()
                    )
                    hidden, aux = unpack(packed, hidden.shape)
                else:
                    hidden = checkpoint(
                        block.hidden, hidden, parameters=block.parameters()
                    )
                    aux = Tensor(
                        get_backend(hidden.device).full(1, dtype=hidden.dtype), shape=()
                    )
            else:
                hidden, aux = block(hidden)
            auxiliary.append(aux)
        aux = auxiliary[0]
        for value in auxiliary[1:]:
            aux = aux + value
        self.last_auxiliary_loss = aux / self.config.n_layers
        return self.norm(hidden)

    def __call__(self, tokens, *, train=True):
        with set_recording(train), set_training(train):
            hidden = self.hidden(tokens, train=train)
            # 输出头复用同一个 embedding 叶子参数，反向自动合并两条贡献。
            return linear(hidden, self.embedding.weight)

    def loss(self, tokens, labels=None, *, train=True, chunk_size=128):
        if chunk_size <= 0:
            raise ValueError("输出头分块大小必须大于零")
        _, shape = self._tokens(tokens)
        if shape[1] < 2:
            raise ValueError("下一字符训练至少需要两个位置")
        labels = tokens if labels is None else labels
        if infer_shape(labels) != shape:
            raise ValueError("监督标签形状必须与输入一致")
        identifiers = tuple(flatten_indices(labels))
        if any(
            (
                label != -100 and (not 0 <= label < self.config.vocab_size)
                for label in identifiers
            )
        ):
            raise ValueError("标签必须是词表编号或 -100")
        with set_recording(train), set_training(train):
            # 标签在 Transformer 计算之前上传一次；输出头前后向复用同一缓冲。
            prepared_labels = backend(self.embedding.weight).prepare_labels(
                tuple(label for row in labels for label in row[1:])
            )
            hidden = self.hidden(tokens, train=train)
            self.last_lm_loss = projection_loss(
                hidden,
                self.embedding.weight,
                labels,
                chunk_size,
                prepared_labels=prepared_labels,
            )
            return (
                self.last_lm_loss + self.config.aux_loss_coef * self.last_auxiliary_loss
            )
