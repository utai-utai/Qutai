from . import apple, cpu, cuda
from .storage import Storage


_BACKENDS = {"cpu": cpu, "cuda": cuda, "apple": apple}


def get_backend(device: str | Storage = "cpu", *others: Storage):
    """按设备或存储选择后端；多个存储必须位于同一设备并使用相同精度。"""
    if isinstance(device, Storage):
        if any((item.device, item.dtype) != (device.device, device.dtype) for item in others):
            raise ValueError("参与运算的设备和精度必须相同")
        device = device.device
    if device not in _BACKENDS: raise ValueError(f"尚不支持设备：{device}")
    return _BACKENDS[device]


def default_dtype(device: str, *, complex_data: bool = False) -> str:
    """CPU 默认双精度，GPU 默认单精度；不触发设备初始化。"""
    get_backend(device)
    if complex_data: return "complex128" if device == "cpu" else "complex64"
    return "float64" if device == "cpu" else "float32"


def is_available(device: str) -> bool:
    """尚未实现或本机不可用的设备返回 False。"""
    return device in _BACKENDS and _BACKENDS[device].is_available()


def synchronize(device: str = "cpu") -> None:
    """等待指定设备此前提交的计算完成。"""
    get_backend(device).synchronize()


def transfer(values: Storage, device: str | None = None, dtype: str | None = None, *, copy: bool = False) -> Storage:
    """跨设备采用目标默认精度；同设备保留精度，显式 dtype 优先。"""
    device = values.device if device is None else device
    if dtype is None:
        dtype = (values.dtype if device == values.device or values.dtype == "int64"
                 else default_dtype(device, complex_data=values.dtype.startswith("complex")))
    backend = get_backend(device)
    if (device, dtype) == (values.device, values.dtype) and not copy: return values
    if device == values.device == "cuda": return backend.from_storage(values, dtype=dtype)
    get_backend(values.device).synchronize()
    result = backend.from_storage(values, dtype=dtype)
    backend.synchronize()
    return result
