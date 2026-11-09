"""教程数据：Iris 读取 CSV，MNIST 保留原始压缩文件，图片按需转为 Tensor。"""

from csv import reader
from gzip import decompress
from pathlib import Path
from ssl import create_default_context
from struct import unpack
from urllib.request import urlopen

from qutai import Tensor


DIRECTORY = Path(__file__).parent
IRIS_NAMES = ("setosa", "versicolor", "virginica")


def load_iris() -> tuple[list[list[float]], list[int]]:
    """返回四个花朵特征，以及 0/1/2 三种类别编号。"""
    with (DIRECTORY / "iris.csv").open(newline="", encoding="utf-8") as file:
        rows = list(reader(file))[1:]
    return [[float(value) for value in row[:4]] for row in rows], [IRIS_NAMES.index(row[4]) for row in rows]


class Images:
    """只保存原始像素；images[i] 返回 (1, 28, 28) 的 0–1 图片。"""
    def __init__(self, pixels: bytes) -> None:
        self.pixels = pixels

    def __len__(self) -> int:
        return len(self.pixels) // 784

    def __getitem__(self, index: int | slice) -> Tensor | list[Tensor]:
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError("图片索引超出范围")
        return Tensor([pixel / 255 for pixel in self.pixels[index * 784:(index + 1) * 784]], shape=(1, 28, 28))


def _read(name: str) -> bytes:
    """首次下载，之后直接读取 examples/data/mnist 中的缓存。"""
    path = DIRECTORY / "mnist" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        certificates = Path("/etc/ssl/cert.pem")
        context = create_default_context(cafile=str(certificates) if certificates.exists() else None)
        with urlopen("https://storage.googleapis.com/cvdf-datasets/mnist/" + name, context=context, timeout=60) as response:
            path.write_bytes(response.read())
    return decompress(path.read_bytes())


def load_mnist(split: str = "train") -> tuple[Images, bytes]:
    """返回可索引的图片与标签，split 为 train 或 test。"""
    prefix = {"train": "train", "test": "t10k"}[split]
    images = _read(prefix + "-images-idx3-ubyte.gz")
    targets = _read(prefix + "-labels-idx1-ubyte.gz")
    magic, count, height, width = unpack(">IIII", images[:16])
    label_magic, label_count = unpack(">II", targets[:8])
    if (magic, label_magic, height, width) != (2051, 2049, 28, 28) or count != label_count:
        raise ValueError("MNIST 图片与标签的格式或数量不匹配")
    if len(images) != 16 + count * 784 or len(targets) != 8 + count:
        raise ValueError("MNIST 文件不完整")
    return Images(images[16:]), targets[8:]
