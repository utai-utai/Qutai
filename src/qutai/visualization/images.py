from base64 import b64encode
from collections.abc import Iterable
from html import escape
from pathlib import Path
from struct import pack
from zlib import compress, crc32

from ..core.tensor import Tensor


def _chunk(kind: bytes, content: bytes) -> bytes:
    """PNG 数据块由长度、类型、内容与 CRC 组成。"""
    return pack(">I", len(content)) + kind + content + pack(">I", crc32(kind + content))


class Image:
    """接收 CHW 像素字节，Notebook 自动显示为最近邻放大的 PNG。"""
    def __init__(self, pixels: bytes, shape: tuple[int, int, int] = (1, 28, 28), *, scale: int = 6) -> None:
        channels, height, width = shape
        if channels not in (1, 3): raise ValueError("图像只支持 1 个灰度通道或 3 个 RGB 通道")
        if min(height, width, scale) <= 0: raise ValueError("图像尺寸和显示倍数必须大于零")
        if len(pixels) != channels * height * width: raise ValueError("像素字节数量与图像形状不一致")
        self.pixels = pixels
        self.shape = shape
        self.scale = scale

    @staticmethod
    def from_tensor(tensor: Tensor, *, value_range: tuple[float, float] = (-1, 1), scale: int = 6) -> Image:
        """将 CHW Tensor 映射为显示像素，只裁剪显示结果，不修改原始数据。"""
        if len(tensor.shape) != 3: raise ValueError("显示图像需要（通道，高，宽）三维 Tensor")
        lower, upper = value_range
        if upper <= lower: raise ValueError("显示范围的上限必须大于下限")
        pixels = bytes(round(255 * max(0.0, min(1.0, (value - lower) / (upper - lower)))) for value in tensor.storage.tolist())
        channels, height, width = tensor.shape
        return Image(pixels, (channels, height, width), scale=scale)

    def _repr_png_(self) -> bytes:
        """CHW 转换为 PNG 的逐行像素，按整数倍复制像素和扫描行。"""
        channels, height, width = self.shape
        rows = bytearray()
        for y in range(height):
            row = bytes(self.pixels[channel * height * width + y * width + x] for x in range(width) for channel in range(channels))
            scaled = b"".join(row[x * channels:(x + 1) * channels] * self.scale for x in range(width))
            rows.extend((b"\x00" + scaled) * self.scale)
        color = 0 if channels == 1 else 2
        header = pack(">IIBBBBB", width * self.scale, height * self.scale, 8, color, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", compress(rows)) + _chunk(b"IEND", b"")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self._repr_png_())
        return path


class ImageCombine:
    """将多张图像组合显示；PNG 内嵌在 HTML 中，可单独保存。"""
    def __init__(self, images: Iterable[Image], *, columns: int = 4, labels: Iterable[object] | None = None, title: str = "") -> None:
        if columns <= 0: raise ValueError("图像网格的列数必须大于零")
        self.images = list(images)
        self.labels = [None] * len(self.images) if labels is None else list(labels)
        if len(self.labels) != len(self.images): raise ValueError("图像标签数量必须与图像数量相同")
        self.columns = columns
        self.title = title

    def _repr_html_(self) -> str:
        body = '<section style="overflow-x:auto;font:14px system-ui,sans-serif;color:#1e293b;background:white;padding:12px">'
        if self.title: body += f'<h3 style="font-size:16px;margin:8px 0 14px">{escape(str(self.title))}</h3>'
        if not self.images: return body + '<p>暂无图像</p></section>'
        body += f'<div style="display:grid;grid-template-columns:repeat({self.columns},max-content);gap:12px;align-items:start">'
        for index, (image, label) in enumerate(zip(self.images, self.labels, strict=True)):
            encoded = b64encode(image._repr_png_()).decode("ascii")
            alt = f"图像 {index + 1}" if label is None else str(label)
            body += f'<figure style="margin:0"><img src="data:image/png;base64,{encoded}" alt="{escape(alt, quote=True)}" style="display:block;image-rendering:pixelated;border:1px solid #cbd5e1">'
            if label is not None: body += f'<figcaption style="padding-top:6px;text-align:center">{escape(str(label))}</figcaption>'
            body += '</figure>'
        return body + '</div></section>'

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        content = f'<!doctype html><html><head><meta charset="utf-8"><title>{escape(str(self.title))}</title></head><body>{self._repr_html_()}</body></html>'
        path.write_text(content, encoding="utf-8")
        return path
