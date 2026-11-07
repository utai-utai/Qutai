import gzip
import struct
import sys
import tarfile
from collections.abc import Sequence
from math import prod
from pathlib import Path
from ssl import create_default_context
from urllib.request import urlopen

from qutai import Tensor


DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_DIRECTORY = DATA_DIRECTORY / "mnist"
MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
CIFAR10_URL = "https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_DIRECTORY = DATA_DIRECTORY / "cifar10"
CIFAR10_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class Images(Sequence[Tensor]):
    """示例使用的图片集合：保留原始像素，按索引取出带形状的归一化图片。"""
    def __init__(self, pixels: bytes, shape: tuple[int, int, int]) -> None:
        self.pixels, self.shape = pixels, shape
        self.size = prod(shape)

    def __len__(self) -> int:
        return len(self.pixels) // self.size

    def __getitem__(self, index: int | slice) -> Tensor | list[Tensor]:
        if isinstance(index, slice): return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError("图片索引超出范围")
        start = index * self.size
        return Tensor([pixel / 255 for pixel in self.pixels[start:start + self.size]], shape=self.shape)


def _download(url: str, path: Path) -> None:
    """原始文件只下载一次；macOS 使用系统 CA，保留 TLS 验证。"""
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("下载数据:", path.name)
    context = create_default_context(cafile="/etc/ssl/cert.pem" if sys.platform == "darwin" else None)
    with urlopen(url, context=context, timeout=60) as response:
        path.write_bytes(response.read())


def load_mnist(split: str = "train", directory: str | Path = MNIST_DIRECTORY) -> tuple[Images, bytes]:
    """读取原始 IDX.gz 缓存，返回可索引的图片和标签字节。"""
    if split not in MNIST_FILES: raise ValueError("MNIST 划分只能是 train 或 test")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contents = []
    for name in MNIST_FILES[split]:
        path = directory / name
        _download(MNIST_URL + name, path)
        contents.append(gzip.decompress(path.read_bytes()))

    images, targets = contents
    if len(images) < 16 or len(targets) < 8: raise ValueError("MNIST 文件的 IDX 头部不完整")
    image_magic, count, height, width = struct.unpack(">IIII", images[:16])
    target_magic, target_count = struct.unpack(">II", targets[:8])
    if image_magic != 2051 or target_magic != 2049: raise ValueError("MNIST 文件的 IDX 类型无效")
    if (height, width) != (28, 28): raise ValueError("MNIST 图像形状必须是 28×28")
    images, targets = images[16:], targets[8:]
    if count != target_count or len(images) != count * height * width or len(targets) != count:
        raise ValueError("MNIST 图像和标签的记录数量不一致")
    if any(target > 9 for target in targets): raise ValueError("MNIST 标签必须是 0 到 9 的数字")
    return Images(images, (1, height, width)), targets


def load_cifar10(split: str = "train", directory: str | Path = CIFAR10_DIRECTORY) -> tuple[Images, bytes]:
    """读取 CIFAR-10 binary，返回可索引的 CHW 图片和类别字节。"""
    if split not in ("train", "test"): raise ValueError("CIFAR-10 划分只能是 train 或 test")
    path = Path(directory) / "cifar-10-binary.tar.gz"
    _download(CIFAR10_URL, path)
    names = [f"data_batch_{index}.bin" for index in range(1, 6)] if split == "train" else ["test_batch.bin"]
    images, targets = bytearray(), bytearray()
    with tarfile.open(path, "r:gz") as archive:
        for name in names:
            member = archive.extractfile("cifar-10-batches-bin/" + name)
            if member is None: raise ValueError("CIFAR-10 压缩包中缺少图像批次")
            with member:
                content = member.read()
            if len(content) % 3073: raise ValueError("CIFAR-10 记录必须是一个标签和 3072 个像素字节")
            pixels = memoryview(content)
            for start in range(0, len(content), 3073):
                target = content[start]
                if target > 9: raise ValueError("CIFAR-10 标签必须是 0 到 9 的数字")
                targets.append(target)
                images.extend(pixels[start + 1:start + 3073])
    return Images(bytes(images), (3, 32, 32)), bytes(targets)
