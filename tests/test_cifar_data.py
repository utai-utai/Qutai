"""CIFAR binary 读取与通用 PNG 显示；仅使用小型离线数据。"""

import io
import struct
import tarfile
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.datasets import CIFAR10_NAMES, CIFAR10_URL, load_cifar10
from qutai.visualization import Image
from qutai import Stream


def archive_bytes(members: dict[str, bytes]) -> bytes:
    """创建只含测试记录的 binary 压缩包。"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def image_record(target: int, red: int, green: int, blue: int) -> bytes:
    """单个 CIFAR 记录：标签后依次是 R、G、B 三个 32×32 平面。"""
    return bytes([target]) + bytes([red]) * 1024 + bytes([green]) * 1024 + bytes([blue]) * 1024


def png_chunks(content: bytes) -> list[tuple[bytes, bytes]]:
    """从 PNG 字节检查块边界和 CRC，供像素验证使用。"""
    if content[:8] != b"\x89PNG\r\n\x1a\n": raise ValueError("PNG 签名无效")
    result = []
    offset = 8
    while offset < len(content):
        size = struct.unpack(">I", content[offset:offset + 4])[0]
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:offset + 8 + size]
        checksum = struct.unpack(">I", content[offset + 8 + size:offset + 12 + size])[0]
        if checksum != zlib.crc32(kind + data): raise ValueError("PNG CRC 无效")
        result.append((kind, data))
        offset += size + 12
    return result


class CIFARDataTests(unittest.TestCase):
    def test_training_members_preserve_chw_and_do_not_extract_files(self):
        members = {f"cifar-10-batches-bin/data_batch_{index}.bin": image_record(index - 1, index, index + 10, index + 20) for index in range(1, 6)}
        members["cifar-10-batches-bin/test_batch.bin"] = b"unused invalid record"
        members["../../ignored.txt"] = b"do not extract"
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cifar-10-binary.tar.gz"
            path.write_bytes(archive_bytes(members))
            with patch("examples.datasets.urlopen") as download:
                images, targets = load_cifar10(directory=folder)
                download.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [path])
        self.assertEqual(targets, bytes(range(5)))
        self.assertEqual(len(images), 5)
        self.assertEqual(images[0].shape, (3, 32, 32))
        self.assertEqual(images.pixels[:3072], image_record(0, 1, 11, 21)[1:])
        inputs, labels = next(iter(Stream(images, targets, batch_size=2, progress=False, shuffle=False)))
        self.assertEqual(inputs.shape, (2, 3, 32, 32))
        self.assertEqual(labels, [0, 1])
        self.assertEqual(inputs.values[:1024], [1 / 255] * 1024)
        self.assertEqual(inputs.values[1024:2048], [11 / 255] * 1024)
        self.assertEqual(inputs.values[2048:3072], [21 / 255] * 1024)

    def test_test_split_downloads_once_and_preserves_archive(self):
        record = image_record(9, 255, 128, 0)
        content = archive_bytes({"cifar-10-batches-bin/test_batch.bin": record})
        with TemporaryDirectory() as folder:
            directory = Path(folder) / "cifar10"
            with patch("examples.datasets.urlopen", return_value=io.BytesIO(content)) as download:
                first = load_cifar10("test", directory)
                second = load_cifar10("test", directory)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.args[0], CIFAR10_URL)
            self.assertEqual((directory / "cifar-10-binary.tar.gz").read_bytes(), content)
        self.assertEqual((first[0].pixels, first[1]), (record[1:], bytes([9])))
        self.assertEqual((second[0].pixels, second[1]), (first[0].pixels, first[1]))
        self.assertEqual(CIFAR10_NAMES[0], "airplane")
        self.assertEqual(CIFAR10_NAMES[9], "truck")

    def test_invalid_split_record_length_and_label(self):
        with self.assertRaisesRegex(ValueError, "train 或 test"):
            load_cifar10("unknown")
        for record in (b"too short", image_record(10, 0, 0, 0)):
            with self.subTest(size=len(record)):
                with TemporaryDirectory() as folder:
                    path = Path(folder) / "cifar-10-binary.tar.gz"
                    path.write_bytes(archive_bytes({"cifar-10-batches-bin/test_batch.bin": record}))
                    with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                        load_cifar10("test", folder)


class ImageDisplayTests(unittest.TestCase):
    def test_grayscale_png_dimensions_crc_and_pixels(self):
        chunks = png_chunks(Image(bytes([0, 64, 128, 255]), (1, 2, 2), scale=1)._repr_png_())
        self.assertEqual([kind for kind, _ in chunks], [b"IHDR", b"IDAT", b"IEND"])
        header = struct.unpack(">IIBBBBB", chunks[0][1])
        self.assertEqual(header, (2, 2, 8, 0, 0, 0, 0))
        self.assertEqual(zlib.decompress(chunks[1][1]), bytes([0, 0, 64, 0, 128, 255]))

    def test_rgb_chw_is_interleaved_and_nearest_neighbor_scaled(self):
        pixels = bytes([255, 0, 0, 255, 64, 128])
        chunks = png_chunks(Image(pixels, (3, 1, 2), scale=2)._repr_png_())
        header = struct.unpack(">IIBBBBB", chunks[0][1])
        self.assertEqual(header, (4, 2, 8, 2, 0, 0, 0))
        row = bytes([0, 255, 0, 64, 255, 0, 64, 0, 255, 128, 0, 255, 128])
        self.assertEqual(zlib.decompress(chunks[1][1]), row * 2)
        self.assertEqual(pixels, bytes([255, 0, 0, 255, 64, 128]))

    def test_grayscale_uses_same_scaling_interface(self):
        chunks = png_chunks(Image(bytes([0, 255]), (1, 1, 2), scale=2)._repr_png_())
        self.assertEqual(zlib.decompress(chunks[1][1]), bytes([0, 0, 0, 255, 255]) * 2)

    def test_invalid_image_shapes_and_scale(self):
        cases = ((b"", (1, 1, 1), 1), (bytes(2), (2, 1, 1), 1), (b"", (1, 0, 1), 1), (bytes(1), (1, 1, 1), 0))
        for pixels, shape, scale in cases:
            with self.subTest(shape=shape, scale=scale):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    Image(pixels, shape, scale=scale)


if __name__ == "__main__":
    unittest.main()
