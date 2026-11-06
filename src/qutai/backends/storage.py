from typing import Self


class Storage:
    """数值存储的共同约定；不依赖 Tensor 或具体设备。"""
    device: str
    dtype: str

    @property
    def size(self) -> int:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    def tolist(self) -> list[float] | list[complex]:
        raise NotImplementedError
