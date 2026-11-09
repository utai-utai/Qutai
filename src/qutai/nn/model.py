from pathlib import Path

from ..core.context import set_recording, set_training
from ..core.migration import move
from ..core.tensor import Tensor
from ..core.types import Data
from .layers import Sequence
from .serialization import load_into, restore_layers, save_model


class Model(Sequence):
    """按顺序执行各层，支持嵌套模型。"""
    def __call__(self, data: Data | Tensor, *, train: bool = True) -> Tensor:
        """设置各层的训练状态；预测时同时暂停计算图记录。"""
        with set_recording(train), set_training(train):
            tensor = super().__call__(data)
        if not train and tensor.requires_grad: return Tensor(tensor.storage.copy(), shape=tensor.shape)
        return tensor

    def save(self, path: str | Path) -> None:
        """将模型结构和参数一起保存为 JSON 文件。"""
        save_model(self, path, Model)

    def load(self, path: str | Path) -> Model:
        """空模型恢复结构与参数，已有结构时原地载入参数；返回自身。"""
        if self.layers: load_into(self, path, Model)
        else:
            self.layers = restore_layers(path, Model)
            move(self, self._device, self._dtype)
        return self
