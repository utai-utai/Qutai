from collections.abc import Iterable
from math import prod
from random import Random
from statistics import NormalDist
from typing import Self

from ..backends import cuda
from ..backends.device import get_backend
from ..backends.storage import Storage
from ..core.context import is_training, set_recording
from ..core.migration import move
from ..core.shape import normalize_axis, pair
from ..core.tensor import Tensor, add, batch_norm2d, conv2d, embedding, gelu, layer_norm, matmul, max_pool2d, mean, relu, reshape, softmax
from ..core.types import Data, IntegerData, LayerLike
from .cache import KVCache


class Layer:
    """层的公共行为：计算、提供参数，以及用 + 组合分支。"""
    def __init__(self, *, device: str | None = None, dtype: str | None = None) -> None:
        target = "cpu" if device is None else device
        get_backend(target)
        allowed = ("float32",) if target == "apple" else ("float32", "float64", "bfloat16") if target == "cuda" else ("float32", "float64")
        if dtype is not None and dtype not in allowed: raise ValueError(f"{target} 网络层不支持精度：{dtype}")
        self._device, self._dtype = device, dtype

    def __call__(self, data: Data | Tensor) -> Tensor:
        raise NotImplementedError("层需要实现自己的计算过程")

    def _input(self, data: Data | Tensor) -> Tensor:
        if isinstance(data, Tensor): return data
        parameters = self.parameters()
        reference = parameters[0] if parameters else None
        return Tensor(data, device=getattr(self, "_device", None) if reference is None else reference.device,
                      dtype=getattr(self, "_dtype", None) if reference is None else reference.dtype)

    def parameters(self) -> tuple[Tensor, ...]:
        return ()

    def to(self, device: str | None = None, *, dtype: str | None = None) -> Self:
        """原地迁移参数、梯度与运行统计量，保留参数对象；在前向计算前调用。"""
        move(self, device, dtype)
        if device is not None:
            if device != getattr(self, "_device", None): self._dtype = None
            self._device = device
        if dtype is not None: self._dtype = dtype
        return self

    def buffers(self) -> dict[str, Storage]:
        """需要一起迁移、但不参加参数更新的存储，由具体层显式提供。"""
        return {}

    def __add__(self, other: LayerLike[Tensor]) -> Residual:
        """两边接收同一份输入，再将同形状的输出相加。"""
        return Residual(self, other)


class Dropout(Layer):
    def __init__(self, probability=.5, *, seed=None):
        super().__init__()
        if not 0 <= probability < 1: raise ValueError('probability 必须在 [0, 1) 内')
        self.probability, self.random = probability, Random(seed)

    def __call__(self, data, train=None):
        x = self._input(data)
        if not is_training() or train is False or not self.probability: return x
        keep = 1 - self.probability
        if x.device == 'cuda':
            dtype = 'float32' if x.dtype == 'bfloat16' else x.dtype
            noise = cuda.normal(x.size, mean=-NormalDist().inv_cdf(self.probability), seed=self.random.randrange(2**63), dtype=dtype)
            values = cuda.relu_backward(cuda.full(x.size, 1/keep, dtype=dtype), noise)
            mask = Tensor(values, shape=x.shape).to(dtype=x.dtype)
        else:
            mask = Tensor([float(self.random.random() < keep)/keep for _ in range(x.size)], shape=x.shape, device=x.device, dtype=x.dtype)
        return x * mask


class Sequence(Layer):
    """内部共用的顺序容器，供 Model 和组合层复用。"""
    def __init__(self, layers: Iterable[LayerLike[Tensor]] = (), *, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.layers: tuple[LayerLike[Tensor], ...] = tuple(layers)
        if device is not None or dtype is not None: move(self, device, dtype)

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys(parameter for layer in self.layers for parameter in layer.parameters()))


class Identity(Layer):
    """原样传递输入，不增加计算或参数。"""
    def __call__(self, data: Data | Tensor) -> Tensor:
        return self._input(data)


class Embedding(Layer):
    """接收原始整数编号，查出对应的可学习向量；重复编号共享同一行参数。"""
    def __init__(self, count: int, features: int, *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if count <= 0 or features <= 0: raise ValueError("编号数量和特征数必须大于零")
        random = Random(seed)
        self.weight = Tensor([random.gauss(0.0, 0.02) for _ in range(count * features)], shape=(count, features), requires_grad=True, device=device, dtype=dtype)

    def __call__(self, indices: IntegerData) -> Tensor:
        return embedding(indices, self.weight)

    def parameters(self) -> tuple[Tensor, ...]:
        return (self.weight,)


class Linear(Layer):
    """对输入的最后一轴做线性变换，其余批次和序列轴保持不变。"""
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if in_features <= 0 or out_features <= 0: raise ValueError("输入和输出特征数必须大于零")
        random = Random(seed)
        bound = in_features ** -0.5
        self.weight = Tensor([[random.uniform(-bound, bound) for _ in range(out_features)] for _ in range(in_features)], requires_grad=True, device=device, dtype=dtype)
        self.bias = Tensor([0.0] * out_features, requires_grad=True, device=device, dtype=dtype) if bias else None

    def __call__(self, data: Data | Tensor) -> Tensor:
        """计算 输入 @ 权重 + 偏置，支持向量和任意批次维。"""
        result = matmul(data, self.weight)
        return result if self.bias is None else add(result, self.bias)

    def parameters(self) -> tuple[Tensor, ...]:
        """返回权重和偏置的引用，供优化器更新。"""
        return (self.weight,) if self.bias is None else (self.weight, self.bias)


class Conv2d(Layer):
    """NCHW 二维卷积，权重按（输出通道，输入通道，核高，核宽）保存。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, bias: bool = True, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if in_channels <= 0 or out_channels <= 0: raise ValueError("输入和输出通道数必须大于零")
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        if min(self.kernel_size + self.stride) <= 0: raise ValueError("卷积核尺寸和步长必须大于零")
        if min(self.padding) < 0: raise ValueError("填充长度不能为负数")
        shape = (out_channels, in_channels, *self.kernel_size)
        random = Random(seed)
        bound = (6 / (in_channels * prod(self.kernel_size))) ** 0.5
        self.weight = Tensor([random.uniform(-bound, bound) for _ in range(prod(shape))], shape=shape, requires_grad=True, device=device, dtype=dtype)
        self.bias = Tensor([0.0] * out_channels, requires_grad=True, device=device, dtype=dtype) if bias else None

    def __call__(self, data: Data | Tensor) -> Tensor:
        return conv2d(data, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def parameters(self) -> tuple[Tensor, ...]:
        return (self.weight,) if self.bias is None else (self.weight, self.bias)


class MaxPool2d(Layer):
    """按窗口取最大值，默认步长等于窗口尺寸，没有可训练参数。"""
    def __init__(self, kernel_size: int | tuple[int, int], *, stride: int | tuple[int, int] | None = None, padding: int | tuple[int, int] = 0, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.kernel_size = pair(kernel_size)
        self.stride = self.kernel_size if stride is None else pair(stride)
        self.padding = pair(padding)
        if min(self.kernel_size + self.stride) <= 0: raise ValueError("池化窗口尺寸和步长必须大于零")
        if min(self.padding) < 0: raise ValueError("填充长度不能为负数")

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return max_pool2d(tensor, self.kernel_size, stride=self.stride, padding=self.padding)


class Flatten(Layer):
    """从指定轴展平至末尾，默认保留第一个批次轴。"""
    def __init__(self, start_dim: int = 1, *, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.start_dim = start_dim

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        axis = normalize_axis(self.start_dim, max(len(tensor.shape), 1))
        return reshape(tensor, tensor.shape[:axis] + (prod(tensor.shape[axis:]),))


class BatchNorm2d(Layer):
    """按通道归一化 NCHW 数据；训练使用当前批统计量，预测使用运行统计量。"""
    def __init__(self, num_features: int, *, eps: float = 1e-5, momentum: float = 0.1, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if num_features <= 0: raise ValueError("通道数必须大于零")
        if eps <= 0 or not 0 <= momentum <= 1: raise ValueError("eps 必须大于零，运行统计更新比例必须在 0 到 1 之间")
        self.eps, self.momentum = eps, momentum
        self.weight = Tensor([1.0] * num_features, requires_grad=True, device=device, dtype=dtype)
        self.bias = Tensor([0.0] * num_features, requires_grad=True, device=device, dtype=dtype)
        backend = get_backend(self.weight.device)
        self.running_mean = backend.full(num_features, dtype=self.weight.dtype)
        self.running_var = backend.full(num_features, 1.0, dtype=self.weight.dtype)

    def buffers(self) -> dict[str, Storage]:
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def __call__(self, data: Data | Tensor) -> Tensor:
        return batch_norm2d(data, self.weight, self.bias, self.running_mean, self.running_var,
                            train=is_training(), momentum=self.momentum, eps=self.eps)

    def parameters(self) -> tuple[Tensor, ...]:
        return self.weight, self.bias


class GlobalAvgPool2d(Layer):
    """将每个通道的整张特征图取平均，NCHW 变为 NC。"""
    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        if len(tensor.shape) != 4: raise ValueError("全局平均池化需要（批次，通道，高，宽）四维数据")
        return mean(mean(tensor, axis=-1), axis=-1)


class ReLU(Layer):
    """逐元素保留正数，将其余值置零。"""
    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return relu(tensor)


class GELU(Layer):
    """平滑的逐元素激活，使用标准正态分布的精确定义。"""
    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return gelu(tensor)


class Mean(Layer):
    """沿指定轴取平均，例如 axis=1 将一段序列汇总成一个向量。"""
    def __init__(self, axis: int | None = 1, *, keepdims: bool = False, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.axis, self.keepdims = axis, keepdims

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return tensor.mean(axis=self.axis, keepdims=self.keepdims)


class LayerNorm(Layer):
    """独立归一化每个位置的最后一轴，训练与预测使用相同的计算。"""
    def __init__(self, features: int, *, eps: float = 1e-5, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if features <= 0 or eps <= 0: raise ValueError("特征数和 eps 必须大于零")
        self.eps = eps
        self.weight = Tensor([1.0] * features, requires_grad=True, device=device, dtype=dtype)
        self.bias = Tensor([0.0] * features, requires_grad=True, device=device, dtype=dtype)

    def __call__(self, data: Data | Tensor) -> Tensor:
        return layer_norm(data, self.weight, self.bias, eps=self.eps)

    def parameters(self) -> tuple[Tensor, ...]:
        return self.weight, self.bias


class Softmax(Layer):
    """沿指定轴把各项得分转换为总和为 1 的概率，无可训练参数。"""
    def __init__(self, axis: int = -1, *, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.axis = axis

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return softmax(tensor, self.axis)


class Attention(Layer):
    """单头自注意力：输入（批次，序列长度，特征数），用 Q/K 的相关程度组合 V。"""
    def __init__(self, in_features: int, out_features: int | None = None, *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        out_features = in_features if out_features is None else out_features
        seeds = (None, None, None) if seed is None else (seed, seed + 1, seed + 2)
        self.query = Linear(in_features, out_features, seed=seeds[0], device=device, dtype=dtype)
        self.key = Linear(in_features, out_features, seed=seeds[1], device=device, dtype=dtype)
        self.value = Linear(in_features, out_features, seed=seeds[2], device=device, dtype=dtype)

    def weights(self, data: Data | Tensor) -> Tensor:
        """返回（批次，查询位置，来源位置）的权重，每行和为 1。"""
        tensor = self._input(data)
        if tensor.ndim != 3: raise ValueError("注意力输入必须为（批次，序列长度，特征数）三维数据")
        queries, keys = self.query(tensor), self.key(tensor)
        scores = queries @ keys.transpose((0, 2, 1)) / queries.shape[-1] ** 0.5
        return softmax(scores, axis=-1)

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        return self.weights(tensor) @ self.value(tensor)

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys(parameter for layer in (self.query, self.key, self.value) for parameter in layer.parameters()))


class MultiHeadAttention(Attention):
    """把特征拆成多个头分别关联位置，合并后再做一次输出投影。"""
    def __init__(self, features: int, heads: int = 1, *, causal: bool = False, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        if heads <= 0 or features % heads: raise ValueError("头数必须大于零，且特征数必须能被头数整除")
        super().__init__(features, seed=seed, device=device, dtype=dtype)
        self.heads, self.causal = heads, causal
        self.output = Linear(features, features, seed=None if seed is None else seed + 3, device=device, dtype=dtype)

    def _split(self, tensor: Tensor) -> Tensor:
        """（批次，长度，特征）变为（批次，头数，长度，每头特征）。"""
        batch, length, features = tensor.shape
        return tensor.reshape((batch, length, self.heads, features // self.heads)).transpose((0, 2, 1, 3))

    def weights(self, data: Data | Tensor) -> Tensor:
        """返回（批次，头数，查询位置，来源位置）的注意力权重。"""
        tensor = self._input(data)
        if tensor.ndim != 3: raise ValueError("注意力输入必须为（批次，序列长度，特征数）三维数据")
        queries, keys = self._split(self.query(tensor)), self._split(self.key(tensor))
        return self._weights(queries, keys)

    def _weights(self, queries: Tensor, keys: Tensor) -> Tensor:
        """查询只覆盖当前片段；因果掩码从已有缓存之后的位置开始。"""
        scores = queries @ keys.transpose((0, 1, 3, 2)) / queries.shape[-1] ** 0.5
        if self.causal:
            length, total = queries.shape[2], keys.shape[2]
            offset = total - length
            mask = [[0.0 if column <= offset + row else -float("inf") for column in range(total)] for row in range(length)]
            scores = scores + mask
        return softmax(scores, axis=-1)

    def __call__(self, data: Data | Tensor, *, cache: KVCache | None = None) -> Tensor:
        """普通调用保留反向；传入缓存时只计算当前片段，并暂停建图。"""
        if cache is not None and not self.causal: raise ValueError("KV cache 只支持因果注意力推理")
        with set_recording(cache is None):
            tensor = self._input(data)
            if tensor.ndim != 3: raise ValueError("注意力输入必须为（批次，序列长度，特征数）三维数据")
            queries, keys, values = (self._split(layer(tensor)) for layer in (self.query, self.key, self.value))
            if cache is not None: keys, values = cache.update(keys, values)
            mixed = self._weights(queries, keys) @ values
            merged = mixed.transpose((0, 2, 1, 3)).reshape((tensor.shape[0], tensor.shape[1], self.value.weight.shape[1]))
            return self.output(merged)

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys((*super().parameters(), *self.output.parameters())))


class FeedForward(Sequence):
    """逐位置执行 Linear → GELU → Linear，不混合序列中的不同位置。"""
    def __init__(self, features: int, hidden_features: int | None = None, *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        hidden_features = features * 4 if hidden_features is None else hidden_features
        super().__init__([
            Linear(features, hidden_features, seed=seed),
            GELU(),
            Linear(hidden_features, features, seed=None if seed is None else seed + 1),
        ], device=device, dtype=dtype)


class TransformerBlock(Layer):
    """Pre-Norm Transformer：注意力与前馈网络各自带归一化和残差。"""
    def __init__(self, features: int, heads: int = 1, *, hidden_features: int | None = None, causal: bool = False, eps: float = 1e-5, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.attention_norm = LayerNorm(features, eps=eps, device=device, dtype=dtype)
        self.attention = MultiHeadAttention(features, heads, causal=causal, seed=seed, device=device, dtype=dtype)
        self.feed_forward_norm = LayerNorm(features, eps=eps, device=device, dtype=dtype)
        self.feed_forward = FeedForward(features, hidden_features, seed=None if seed is None else seed + 4, device=device, dtype=dtype)

    def __call__(self, data: Data | Tensor, *, cache: KVCache | None = None) -> Tensor:
        """缓存只用于推理；归一化、注意力和前馈都只处理当前片段。"""
        with set_recording(cache is None):
            tensor = self._input(data)
            tensor = tensor + self.attention(self.attention_norm(tensor), cache=cache)
            return tensor + self.feed_forward(self.feed_forward_norm(tensor))

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys(parameter for layer in (self.attention_norm, self.attention, self.feed_forward_norm, self.feed_forward) for parameter in layer.parameters()))


class PatchEmbedding(Layer):
    """将 NCHW 图片分成不重叠的 patch，投影为（批次，patch 数，特征）。"""
    def __init__(self, in_channels: int, features: int, patch_size: int | tuple[int, int], *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.patch_size = pair(patch_size)
        self.projection = Conv2d(in_channels, features, self.patch_size, stride=self.patch_size, seed=seed, device=device, dtype=dtype)

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        if tensor.ndim != 4: raise ValueError("图片输入必须为（批次，通道，高，宽）四维数据")
        if any(size % patch for size, patch in zip(tensor.shape[2:], self.patch_size, strict=True)): raise ValueError("图片高和宽必须能被 patch 尺寸整除")
        projected = self.projection(tensor)
        batch, features, height, width = projected.shape
        return projected.reshape((batch, features, height * width)).transpose((0, 2, 1))

    def parameters(self) -> tuple[Tensor, ...]:
        return self.projection.parameters()


class PositionEmbedding(Layer):
    """给序列每个位置加上一个可学习的向量，支持不超过预设长度的序列。"""
    def __init__(self, length: int, features: int, *, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        if length <= 0 or features <= 0: raise ValueError("序列长度和特征数必须大于零")
        random = Random(seed)
        self.weight = Tensor([random.gauss(0.0, 0.02) for _ in range(length * features)], shape=(length, features), requires_grad=True, device=device, dtype=dtype)

    def __call__(self, data: Data | Tensor, *, offset: int = 0) -> Tensor:
        tensor = self._input(data)
        if tensor.ndim != 3 or tensor.shape[-1] != self.weight.shape[-1]: raise ValueError("位置嵌入输入必须为三维数据，且特征数必须匹配")
        if offset < 0 or offset + tensor.shape[1] > self.weight.shape[0]: raise ValueError("输入位置超出位置嵌入的预设长度")
        return tensor + self.weight[offset:offset + tensor.shape[1]]

    def parameters(self) -> tuple[Tensor, ...]:
        return (self.weight,)


class Residual(Layer):
    """主分支加上捷径分支；默认捷径直接保留输入，激活层由外部组合。"""
    def __init__(self, branch: LayerLike[Tensor], shortcut: LayerLike[Tensor] | None = None, *, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.branch = branch
        self.shortcut = shortcut
        if device is not None or dtype is not None: move(self, device, dtype)

    def __call__(self, data: Data | Tensor) -> Tensor:
        tensor = self._input(data)
        main = self.branch(tensor)
        skip = tensor if self.shortcut is None else self.shortcut(tensor)
        if main.shape != skip.shape: raise ValueError("残差两个分支的输出形状必须相同")
        return add(main, skip)

    def parameters(self) -> tuple[Tensor, ...]:
        shortcut = () if self.shortcut is None else self.shortcut.parameters()
        return tuple(dict.fromkeys((*self.branch.parameters(), *shortcut)))


class ResNetBlock(Sequence):
    """两次 Conv/BN 的 ResNet 块，分支相加后接 ReLU，必要时投影捷径。"""
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, seed: int | None = None, device: str | None = None, dtype: str | None = None) -> None:
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        seeds = (None, None, None) if seed is None else (seed, seed + 1, seed + 2)
        branch = Sequence([
            Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, seed=seeds[0]),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, 3, padding=1, bias=False, seed=seeds[1]),
            BatchNorm2d(out_channels),
        ])
        shortcut = Identity() if stride == 1 and in_channels == out_channels else Sequence([
            Conv2d(in_channels, out_channels, 1, stride=stride, bias=False, seed=seeds[2]),
            BatchNorm2d(out_channels),
        ])
        super().__init__([branch + shortcut, ReLU()], device=device, dtype=dtype)
