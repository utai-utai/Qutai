"""CUDA LM 的 BF16 激活；参数、参数梯度、优化器和稳定归约保持 FP32。"""

from contextlib import contextmanager

from ..backends.precision import set_matmul_precision


@contextmanager
def autocast(enabled: bool = True, *, dtype: str = "bfloat16"):
    if dtype != "bfloat16":
        raise ValueError("当前混合精度只支持 bfloat16")
    with set_matmul_precision(dtype if enabled else None):
        yield
