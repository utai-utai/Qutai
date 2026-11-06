from math import cos, exp, log, prod, sin

from ..backends.device import get_backend
from ..core.context import set_recording, set_training
from ..core.shape import pair
from ..core.tensor import Tensor
from ..core.types import Data
from .layers import FeedForward, GELU, Layer, LayerNorm, Linear, MultiHeadAttention, PatchEmbedding, PositionEmbedding, Sequence


class TimeEmbedding(Sequence):
    """将归一化时间的固定正弦、余弦编码送入可学习 MLP；时间本身不求导。"""
    def __init__(self, features: int, *, seed: int | None = None) -> None:
        self.features = features
        super().__init__([
            Linear(features, features * 4, seed=seed), GELU(),
            Linear(features * 4, features, seed=None if seed is None else seed + 1),
        ])

    def __call__(self, times: list[float]) -> Tensor:
        if any(type(time) not in (int, float) or not 0 <= time <= 1 for time in times): raise ValueError("时间必须是 [0, 1] 范围内的实数")
        half = (self.features + 1) // 2
        frequencies = [exp(-log(10000) * index / max(half - 1, 1)) for index in range(half)]
        values = []
        for time in times:
            angles = [time * 1000 * frequency for frequency in frequencies]
            values.extend(([cos(angle) for angle in angles] + [sin(angle) for angle in angles])[:self.features])
        return super().__call__(Tensor(values, shape=(len(times), self.features)))


class DiTBlock(Layer):
    """时间条件生成两组平移、缩放和残差门控，初始时整块保持恒等映射。"""
    def __init__(self, features: int, heads: int = 1, *, hidden_features: int | None = None, seed: int | None = None) -> None:
        self.attention_norm = LayerNorm(features)
        self.attention = MultiHeadAttention(features, heads, seed=seed)
        self.feed_forward_norm = LayerNorm(features)
        self.feed_forward = FeedForward(features, hidden_features, seed=None if seed is None else seed + 4)
        modulation = Linear(features, features * 6, seed=None if seed is None else seed + 6)
        modulation.weight.storage = get_backend(modulation.weight.device).full(modulation.weight.size)
        modulation.bias.storage = get_backend(modulation.bias.device).full(modulation.bias.size)
        self.modulation = Sequence([GELU(), modulation])

    def __call__(self, data: Data | Tensor, condition: Data | Tensor) -> Tensor:
        tensor = data if isinstance(data, Tensor) else Tensor(data)
        condition = condition if isinstance(condition, Tensor) else Tensor(condition)
        if tensor.ndim != 3 or condition.shape != (tensor.shape[0], tensor.shape[-1]): raise ValueError("条件块需要三维序列和每个样本对应的特征向量")
        controls = self.modulation(condition).reshape((tensor.shape[0], 6, tensor.shape[-1]))
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (controls[:, index:index + 1] for index in range(6))
        tensor = tensor + gate_a * self.attention(self.attention_norm(tensor) * (1 + scale_a) + shift_a)
        return tensor + gate_f * self.feed_forward(self.feed_forward_norm(tensor) * (1 + scale_f) + shift_f)

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys(parameter for layer in (self.attention_norm, self.attention, self.feed_forward_norm, self.feed_forward, self.modulation) for parameter in layer.parameters()))


class DiT(Layer):
    """给定 NCHW 图像和每个样本的时间，预测同形状的噪声或速度。"""
    def __init__(self, in_channels: int = 1, image_size: int | tuple[int, int] = 28, *, patch_size: int | tuple[int, int] = 7, features: int = 16, heads: int = 2, depth: int = 1, hidden_features: int | None = None, seed: int | None = None) -> None:
        self.image_shape = (in_channels, *pair(image_size))
        self.patch_size = pair(patch_size)
        if min((*self.image_shape, *self.patch_size)) <= 0 or depth <= 0: raise ValueError("图像、patch 尺寸和条件块数量必须大于零")
        if any(size % patch for size, patch in zip(self.image_shape[1:], self.patch_size, strict=True)): raise ValueError("图片高和宽必须能被 patch 尺寸整除")
        length = prod(size // patch for size, patch in zip(self.image_shape[1:], self.patch_size, strict=True))
        self.patch_embedding = PatchEmbedding(in_channels, features, self.patch_size, seed=seed)
        self.position_embedding = PositionEmbedding(length, features, seed=None if seed is None else seed + 1)
        self.time_embedding = TimeEmbedding(features, seed=None if seed is None else seed + 2)
        self.blocks = tuple(DiTBlock(features, heads, hidden_features=hidden_features, seed=None if seed is None else seed + 4 + index * 7) for index in range(depth))
        self.output_norm = LayerNorm(features)
        self.output = Linear(features, in_channels * prod(self.patch_size), seed=None if seed is None else seed + 4 + depth * 7)
        self.output.weight.storage = get_backend(self.output.weight.device).full(self.output.weight.size)
        self.output.bias.storage = get_backend(self.output.bias.device).full(self.output.bias.size)

    def __call__(self, images: Data | Tensor, times: list[float], *, train: bool = True) -> Tensor:
        tensor = images if isinstance(images, Tensor) else Tensor(images)
        if tensor.ndim != 4 or tensor.shape[1:] != self.image_shape: raise ValueError("输入图片形状与 DiT 配置不一致")
        if len(times) != tensor.shape[0]: raise ValueError("每个样本必须对应一个时间")
        with set_recording(train), set_training(train):
            tokens = self.position_embedding(self.patch_embedding(tensor))
            condition = self.time_embedding(times)
            for block in self.blocks:
                tokens = block(tokens, condition)
            pixels = self.output(self.output_norm(tokens))
            channels, height, width = self.image_shape
            patch_height, patch_width = self.patch_size
            pixels = pixels.reshape((tensor.shape[0], height // patch_height, width // patch_width, channels, patch_height, patch_width))
            return pixels.transpose((0, 3, 1, 4, 2, 5)).reshape(tensor.shape)

    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(dict.fromkeys(parameter for layer in (self.patch_embedding, self.position_embedding, self.time_embedding, *self.blocks, self.output_norm, self.output) for parameter in layer.parameters()))
