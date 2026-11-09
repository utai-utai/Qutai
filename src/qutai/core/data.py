from collections.abc import Iterator, Sequence
from random import Random

from ..backends.device import get_backend
from .tensor import Tensor
from .types import Batch, GenerationMethod
from ..visualization.progress import Progress


class Stream:
    """统一图像、文本与生成训练的数据流；method 指定 Diffusion 或 FlowMatching。"""
    def __init__(self, data: Sequence, targets: Sequence[int] | None = None, *, batch_size: int = 32, context_length: int = 16, method: GenerationMethod[Tensor] | None = None, shuffle: bool = True, seed: int | None = None, limit: int | None = None, progress: bool = False, device: str | None = None, dtype: str | None = None) -> None:
        if type(batch_size) is not int or batch_size <= 0: raise ValueError("每批样本数必须是正整数")
        if limit is not None and limit < 0: raise ValueError("样本数量上限不能为负数")
        if method is not None and targets is not None: raise ValueError("生成模式不需要类别标签")
        if targets is None and method is None:
            if type(context_length) is not int or context_length <= 0: raise ValueError("上下文长度必须是正整数")
            count = max(0, (len(data) - 1) // context_length)
        else:
            if targets is not None and len(data) != len(targets): raise ValueError("数据与标签的样本数量不一致")
            count = len(data)

        self.data, self.targets = data, targets
        self.method = method
        self.device, self.dtype = device, dtype
        self.batch_size, self.context_length = batch_size, context_length
        self.count = count if limit is None else min(limit, count)
        self.shuffle, self.progress = shuffle, progress
        self.random = Random(seed)

    def __len__(self) -> int:
        """一轮中的批次数，保留最后不足一批的数据。"""
        return (self.count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch[Tensor]]:
        """每轮重新遍历；打乱、时间与噪声共用本数据流的局部随机状态。"""
        indices = list(range(self.count))
        if self.shuffle: self.random.shuffle(indices)
        starts = range(0, self.count, self.batch_size)
        for start in Progress(starts, total=len(self)) if self.progress else starts:
            yield self._batch(indices[start:start + self.batch_size])

    def _batch(self, selected: list[int]) -> Batch[Tensor]:
        """生成模式将 0–1 图片映射到 −1–1，再准备输入、时间和目标。"""
        targets = self.targets
        if targets is None and self.method is None:
            length = self.context_length
            offsets = [index * length for index in selected]
            return ([list(self.data[index:index + length]) for index in offsets],
                    [list(self.data[index + 1:index + length + 1]) for index in offsets])
        samples = [self.data[index] for index in selected]
        tensors = [sample.to(self.device, dtype=self.dtype) if isinstance(sample, Tensor) else Tensor(sample, device=self.device, dtype=self.dtype) for sample in samples]
        shape = tensors[0].shape
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("同一批样本的形状必须一致，请先在外部预处理")
        values = get_backend(*(tensor.storage for tensor in tensors)).concatenate(tuple(tensor.storage for tensor in tensors), tuple((tensor.size,) for tensor in tensors))
        inputs = Tensor(values, shape=(len(selected), *shape))
        if self.method is not None: return self.method.prepare(inputs * 2 - 1, self.random)
        return inputs, [targets[index] for index in selected]
