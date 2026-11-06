from collections.abc import Iterator
from pathlib import Path

from ..core.migration import move
from ..core.tensor import Tensor
from ..core.types import Data
from ..nn.dit import DiT
from ..nn.model import Model as NeuralModel
from ..nn.serialization import load_into, restore_layers
from .diffusion import Diffusion, FlowMatching
from .serialization import read_method, save_model


class Model:
    def __init__(self, network: DiT | None = None, *, method: Diffusion | FlowMatching | None = None) -> None:
        self.network = network
        self.method = Diffusion() if network is not None and method is None else method

    def _components(self) -> tuple[DiT, Diffusion | FlowMatching]:
        if self.network is None or self.method is None: raise ValueError("请先创建或加载生成模型")
        return self.network, self.method

    def __call__(self, images: Data | Tensor, times: list[float], *, train: bool = True) -> Tensor:
        """网络返回同形状预测，具体作为噪声或速度由生成方法决定。"""
        network, _ = self._components()
        return network(images, times, train=train)

    def to(self, device: str | None = None, *, dtype: str | None = None) -> Model:
        """原地迁移模型参数、已有梯度与子层的运行统计量。"""
        return move(self, device, dtype)

    def parameters(self) -> tuple[Tensor, ...]:
        return () if self.network is None else self.network.parameters()

    def _states(self, count: int, seed: int | None) -> Iterator[Tensor]:
        network, method = self._components()
        if type(count) is not int or count <= 0: raise ValueError("生成图片数量必须是正整数")
        return method.states(network, (count, *network.image_shape), seed=seed, like=network.parameters()[0])

    def sample(self, *, count: int = 1, seed: int | None = None) -> Tensor:
        """只保留最后的生成结果，采样步数由 method 控制。"""
        states = self._states(count, seed)
        image = next(states)
        for state in states:
            image = state
        return image

    def trajectory(self, *, count: int = 1, seed: int | None = None) -> list[Tensor]:
        """保存初始噪声和每一步结果，用于展示采样过程。"""
        return list(self._states(count, seed))

    def save(self, path: str | Path) -> None:
        """一起保存网络和生成方法，恢复后即可继续训练或采样。"""
        network, method = self._components()
        save_model(network, method, path)

    def load(self, path: str | Path) -> Model:
        """空模型自动恢复；已有模型核对方法和结构后原地更新参数。"""
        method = read_method(path)
        if self.network is not None:
            if self.method is None or self.method.config() != method.config(): raise ValueError("文件中的生成方法与当前模型不一致")
            load_into(NeuralModel([self.network]), path, NeuralModel)
        else:
            layers = restore_layers(path, NeuralModel)
            if len(layers) != 1 or not isinstance(layers[0], DiT): raise ValueError("生成模型文件需要包含一个 DiT 网络")
            self.network, self.method = layers[0], method
        return self
