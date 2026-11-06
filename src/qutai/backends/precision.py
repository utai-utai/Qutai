"""计算精度上下文；不依赖 Tensor 或自动求导状态。"""

from contextvars import ContextVar

_matmul_precision = ContextVar("qutai_matmul_precision", default=None)


def matmul_precision() -> str | None:
    return _matmul_precision.get()


def set_matmul_precision(dtype: str | None):
    return _matmul_precision.set(dtype)
