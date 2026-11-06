from functools import partial
from math import prod

from ..backends.device import get_backend
from ..backends.storage import Storage
from ..core.grad import record
from ..core.shape import flatten_indices, infer_shape
from ..core.tensor import Tensor, mean, square, subtract
from ..core.types import Data, IntegerData


class MSE:
    """可重复调用的均方误差；同形状输入，返回平均损失的标量 Tensor。"""
    def __call__(self, prediction: Data | Tensor, target: Data | Tensor) -> Tensor:
        prediction = prediction if isinstance(prediction, Tensor) else Tensor(prediction, device=target.device if isinstance(target, Tensor) else None, dtype=target.dtype if isinstance(target, Tensor) else None)
        target = target if isinstance(target, Tensor) else Tensor(target, device=prediction.device, dtype=prediction.dtype)
        if prediction.shape != target.shape: raise ValueError(f"预测与目标的形状必须相同，收到 {prediction.shape} 和 {target.shape}")
        return mean(square(subtract(prediction, target)))


class CrossEntropy:
    """最后一轴是类别，整数标签对应前面的批次或序列轴，返回平均损失。"""
    def __call__(self, scores: Data | Tensor, labels: IntegerData) -> Tensor:
        tensor = scores if isinstance(scores, Tensor) else Tensor(scores)
        if tensor.ndim < 2: raise ValueError("交叉熵的得分至少需要二维，最后一轴为类别")
        rows, columns = prod(tensor.shape[:-1]), tensor.shape[-1]
        if rows == 0 or columns == 0: raise ValueError("交叉熵的样本数和类别数必须大于零")
        if infer_shape(labels) != tensor.shape[:-1]: raise ValueError("标签形状必须与得分除类别轴以外的形状相同")
        try:
            labels = tuple(flatten_indices(labels))
        except TypeError:
            raise ValueError("标签必须是类别范围内的整数编号") from None
        if any(not 0 <= label < columns for label in labels): raise ValueError("标签必须是类别范围内的整数编号")
        result = Tensor(get_backend(tensor.device).cross_entropy(tensor.storage, (rows, columns), labels), shape=())
        return record(result, (tensor,), partial(_cross_entropy_backward, scores=tensor, labels=labels))


def _cross_entropy_backward(gradient: Storage, scores: Tensor, labels: tuple[int, ...]) -> tuple[Storage, ...]:
    """输入梯度为 g * (softmax(scores) - 正确类别标记) / 样本数。"""
    rows, columns = len(labels), scores.shape[-1]
    return (get_backend(scores.device).cross_entropy_backward(gradient, scores.storage, (rows, columns), labels),)
