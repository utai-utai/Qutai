from collections.abc import Callable, Iterator
from math import cos, pi, prod, sqrt
from random import Random

from ..core.context import set_recording, set_training
from ..core.tensor import Tensor
from ..core.types import Shape


def normal(shape: Shape, random: Random, like: Tensor | None = None) -> Tensor:
    """使用调用者的局部随机状态，返回不记录梯度的高斯噪声。"""
    return Tensor([random.gauss(0.0, 1.0) for _ in range(prod(shape))], shape=shape, device=None if like is None else like.device, dtype=None if like is None else like.dtype)


def _cosine(time: float) -> float:
    return cos((time + 0.008) / 1.008 * pi / 2) ** 2 / cos(0.008 / 1.008 * pi / 2) ** 2


class Diffusion:
    def __init__(self, steps: int = 16) -> None:
        if type(steps) is not int or steps <= 0: raise ValueError("扩散步数必须是正整数")
        self.steps = steps
        self.betas = [min(1 - _cosine((step + 1) / steps) / _cosine(step / steps), 0.999) for step in range(steps)]
        self.alphas = [1 - beta for beta in self.betas]
        self.alpha_bars = [1.0]
        for alpha in self.alphas:
            self.alpha_bars.append(self.alpha_bars[-1] * alpha)

    def config(self) -> dict[str, str | int]:
        return {"type": "Diffusion", "steps": self.steps, "schedule": "cosine"}

    def prepare(self, images: Tensor, random: Random) -> tuple[Tensor, list[float], Tensor]:
        """供 Stream 调用：每张图随机选择一步，训练目标为加入的噪声。"""
        selected = [random.randint(1, self.steps) for _ in range(images.shape[0])]
        times = [step / self.steps for step in selected]
        noise = normal(images.shape, random, images)
        shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        signal = Tensor([sqrt(self.alpha_bars[step]) for step in selected], shape=shape, device=images.device, dtype=images.dtype)
        scale = Tensor([sqrt(1 - self.alpha_bars[step]) for step in selected], shape=shape, device=images.device, dtype=images.dtype)
        return signal * images + scale * noise, times, noise

    def states(self, model: Callable[..., Tensor], shape: Shape, *, seed: int | None = None, like: Tensor | None = None) -> Iterator[Tensor]:
        """从高斯噪声开始逐步去噪；每次输出前恢复调用者的状态。"""
        random = Random(seed)
        state = normal(shape, random, like)
        yield state
        for step in range(self.steps, 0, -1):
            with set_recording(False), set_training(False):
                predicted = model(state, [step / self.steps] * shape[0], train=False)
                beta, alpha = self.betas[step - 1], self.alphas[step - 1]
                current, previous = self.alpha_bars[step], self.alpha_bars[step - 1]
                state = (state - beta / sqrt(1 - current) * predicted) / sqrt(alpha)
                if step > 1:
                    variance = beta * (1 - previous) / (1 - current)
                    state = state + sqrt(variance) * normal(shape, random, state)
            yield state


class FlowMatching:
    def __init__(self, steps: int = 16) -> None:
        if type(steps) is not int or steps <= 0: raise ValueError("流匹配步数必须是正整数")
        self.steps = steps

    def config(self) -> dict[str, str | int]:
        return {"type": "FlowMatching", "steps": self.steps, "solver": "euler"}

    def prepare(self, images: Tensor, random: Random) -> tuple[Tensor, list[float], Tensor]:
        """供 Stream 调用：从噪声到图像作直线插值，训练目标为恒定速度。"""
        times = [random.random() for _ in range(images.shape[0])]
        noise = normal(images.shape, random, images)
        time = Tensor(times, shape=(images.shape[0],) + (1,) * (images.ndim - 1), device=images.device, dtype=images.dtype)
        return (1 - time) * noise + time * images, times, images - noise

    def states(self, model: Callable[..., Tensor], shape: Shape, *, seed: int | None = None, like: Tensor | None = None) -> Iterator[Tensor]:
        """从 t=0 积分到 t=1；输出中间状态时不保留内部上下文。"""
        random = Random(seed)
        state = normal(shape, random, like)
        yield state
        for step in range(self.steps):
            with set_recording(False), set_training(False):
                velocity = model(state, [step / self.steps] * shape[0], train=False)
                state = state + velocity / self.steps
            yield state
