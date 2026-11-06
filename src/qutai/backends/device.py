from . import cpu
from .storage import Storage


def get_backend(device: str | Storage = "cpu", *others: Storage):
    """按设备或存储选择后端；多个存储必须位于同一设备并使用相同精度。"""
    if isinstance(device, Storage):
        if any((item.device, item.dtype) != (device.device, device.dtype) for item in others):
            raise ValueError("参与运算的设备和精度必须相同")
        device = device.device
    if device != "cpu": raise ValueError(f"尚不支持设备：{device}")
    return cpu
