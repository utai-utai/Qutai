from ..backends.device import get_backend
from ..core.tensor import Tensor


def _append(previous: Tensor | None, current: Tensor) -> Tensor:
    """沿（批次，头数，长度，特征）的长度轴追加，保存独立的数值。"""
    if previous is None: return Tensor(current.storage.copy(), shape=current.shape)
    batch, heads, length, features = current.shape
    values = get_backend(previous.storage, current.storage).concatenate((previous.storage, current.storage), (previous.shape, current.shape), axis=2)
    return Tensor(values, shape=(batch, heads, previous.shape[2] + length, features))


class KVCache:
    """单层推理期间已经算过的 K、V；不保存计算图或模型参数。"""
    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """追加当前片段的 K、V，返回全部已缓存位置。"""
        if keys.ndim != 4 or keys.shape != values.shape: raise ValueError("K、V 缓存必须为同形状的四维数据")
        if self.keys is not None and self.keys.shape[:2] + self.keys.shape[3:] != keys.shape[:2] + keys.shape[3:]:
            raise ValueError("新增 K、V 的批次、头数和每头特征数必须与缓存一致")
        self.keys, self.values = _append(self.keys, keys), _append(self.values, values)
        return self.keys, self.values
