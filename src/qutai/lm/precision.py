from contextlib import contextmanager

from ..backends.precision import set_matmul_precision


@contextmanager
def autocast(enabled: bool = True, *, dtype: str = "bfloat16"):
    if dtype != "bfloat16":
        raise ValueError("当前混合精度只支持 bfloat16")
    with set_matmul_precision(dtype if enabled else None):
        yield
