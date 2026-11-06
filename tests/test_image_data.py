"""字节图像分批、MNIST gzip 缓存和 IDX 解析；不联网。"""

import gzip
import io
import random
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.datasets import Images, MNIST_FILES, MNIST_URL, load_mnist
from qutai import Stream, Tensor


class ImageBatchTests(unittest.TestCase):
    def test_batch_shape_pixels_labels_and_last_batch(self):
        pixels = bytes([0, 255, 128, 64, 255, 0, 32, 16, 1, 2, 3, 4])
        images = Images(pixels, (1, 2, 2))
        result = list(Stream(images, (3, 5, 8), batch_size=2, shuffle=False, progress=False))
        first, targets = result[0]
        self.assertEqual(first.shape, (2, 1, 2, 2))
        self.assertEqual(first.values, [pixel / 255 for pixel in pixels[:8]])
        self.assertEqual(targets, [3, 5])
        self.assertEqual(result[1][0].shape, (1, 1, 2, 2))
        self.assertEqual(result[1][1], [8])
        self.assertFalse(first.requires_grad)

    def test_shuffle_keeps_fixed_subset_and_random_state(self):
        state = random.getstate()
        images = Images(bytes(range(10)), (1, 1, 1))
        first = list(Stream(images, tuple(range(10)), batch_size=2, limit=5, seed=42, progress=False))
        second = list(Stream(images, tuple(range(10)), batch_size=2, limit=5, seed=42, progress=False))
        self.assertEqual([targets for _, targets in first], [targets for _, targets in second])
        self.assertEqual(sorted(target for _, targets in first for target in targets), list(range(5)))
        self.assertEqual(random.getstate(), state)
        for inputs, targets in first:
            self.assertEqual(inputs.values, [target / 255 for target in targets])

    def test_only_current_batch_is_converted(self):
        images = Images(bytes(range(20)), (1, 1, 1))
        with patch("examples.datasets.Tensor", side_effect=Tensor) as constructor:
            data = Stream(images, tuple(range(20)), batch_size=3, shuffle=False, progress=False)
            constructor.assert_not_called()
            inputs, targets = next(iter(data))
            self.assertEqual(constructor.call_count, 3)
            self.assertEqual([len(call.args[0]) for call in constructor.call_args_list], [1, 1, 1])
            self.assertEqual(inputs.shape, (3, 1, 1, 1))
            self.assertEqual(targets, [0, 1, 2])

    def test_empty_limit_and_invalid_inputs(self):
        self.assertEqual(list(Stream([], [])), [])
        self.assertEqual(list(Stream([Tensor([1])], [0], limit=0)), [])
        for options in ({"batch_size": 0}, {"limit": -1}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    Stream([Tensor([1])], [0], **options)
        with self.assertRaisesRegex(ValueError, "样本数量不一致"):
            Stream([Tensor([1])], [])

    def test_image_collection_indexes_slices_and_bounds(self):
        images = Images(bytes(range(12)), (1, 2, 2))
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].shape, (1, 2, 2))
        self.assertEqual(images[-1].values, [value / 255 for value in range(8, 12)])
        self.assertEqual([image.values[0] for image in images[::2]], [0.0, 8 / 255])
        for index in (-4, 3):
            with self.assertRaises(IndexError): images[index]


class MNISTTests(unittest.TestCase):
    def make_idx(self) -> tuple[bytes, bytes]:
        pixels = bytes(position % 256 for position in range(2 * 784))
        images = struct.pack(">IIII", 2051, 2, 28, 28) + pixels
        targets = struct.pack(">II", 2049, 2) + bytes([0, 9])
        return images, targets

    def write_cache(self, directory: Path, images: bytes, targets: bytes, split: str = "train") -> None:
        for name, content in zip(MNIST_FILES[split], (images, targets), strict=True):
            (directory / name).write_bytes(gzip.compress(content))

    def test_cached_idx_keeps_bytes_and_returns_shaped_samples_without_network(self):
        images, targets = self.make_idx()
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write_cache(directory, images, targets)
            with patch("examples.datasets.urlopen") as download:
                pixels, labels = load_mnist(directory=directory)
                download.assert_not_called()
            self.assertEqual({path.name for path in directory.iterdir()}, set(MNIST_FILES["train"]))
        self.assertIsInstance(pixels, Images)
        self.assertIsInstance(pixels.pixels, bytes)
        self.assertEqual(len(pixels), 2)
        self.assertEqual(pixels[0].shape, (1, 28, 28))
        self.assertIsInstance(labels, bytes)
        self.assertEqual(pixels.pixels, images[16:])
        self.assertEqual(pixels[0].values, [pixel / 255 for pixel in images[16:800]])
        self.assertEqual(labels, targets[8:])

    def test_train_and_test_use_separate_caches(self):
        images, targets = self.make_idx()
        test_images = images[:16] + images[16:][::-1]
        test_targets = targets[:8] + bytes([8, 1])
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write_cache(directory, images, targets)
            self.write_cache(directory, test_images, test_targets, split="test")
            with patch("examples.datasets.urlopen") as download:
                test_pictures, test_labels = load_mnist("test", directory)
                train_pictures, train_labels = load_mnist("train", directory)
                self.assertEqual((test_pictures.pixels, test_labels), (test_images[16:], test_targets[8:]))
                self.assertEqual((train_pictures.pixels, train_labels), (images[16:], targets[8:]))
                download.assert_not_called()
            self.assertEqual({path.name for path in directory.iterdir()},
                             set(MNIST_FILES["train"] + MNIST_FILES["test"]))

    def test_downloads_keep_gzip_cache_and_second_load_reuses_it(self):
        contents = self.make_idx()
        compressed = [gzip.compress(content) for content in contents]
        with TemporaryDirectory() as folder:
            directory = Path(folder) / "mnist"
            responses = [io.BytesIO(content) for content in compressed]
            with patch("examples.datasets.urlopen", side_effect=responses) as download:
                first = load_mnist(directory=directory)
                modified = [(directory / name).stat().st_mtime_ns for name in MNIST_FILES["train"]]
                second = load_mnist(directory=directory)
                self.assertEqual((first[0].pixels, first[1]), (second[0].pixels, second[1]))
                self.assertEqual((first[0].pixels, first[1]), (contents[0][16:], contents[1][8:]))
                self.assertEqual([(directory / name).stat().st_mtime_ns for name in MNIST_FILES["train"]], modified)
                self.assertEqual(download.call_count, 2)
                self.assertEqual([call.args[0] for call in download.call_args_list], [MNIST_URL + name for name in MNIST_FILES["train"]])
            for name, content in zip(MNIST_FILES["train"], compressed, strict=True):
                self.assertEqual((directory / name).read_bytes(), content)

    def test_macos_download_uses_verified_system_ca_context(self):
        responses = [io.BytesIO(gzip.compress(content)) for content in self.make_idx()]
        with TemporaryDirectory() as folder:
            with patch("examples.datasets.sys.platform", "darwin"), patch("examples.datasets.create_default_context") as context, patch("examples.datasets.urlopen", side_effect=responses) as download:
                load_mnist(directory=folder)
                context.assert_called_with(cafile="/etc/ssl/cert.pem")
                self.assertEqual(download.call_count, 2)
                for call in download.call_args_list:
                    self.assertIs(call.kwargs["context"], context.return_value)

    def test_invalid_idx_headers_and_counts(self):
        images, targets = self.make_idx()
        cases = (
            (b"", targets),
            (struct.pack(">IIII", 2049, 2, 28, 28) + images[16:], targets),
            (images, struct.pack(">II", 2051, 2) + targets[8:]),
            (struct.pack(">IIII", 2051, 2, 14, 56) + images[16:], targets),
            (images, struct.pack(">II", 2049, 3) + targets[8:]),
            (images[:-1], targets),
            (images, targets[:-1]),
        )
        for invalid_images, invalid_targets in cases:
            with self.subTest(image_size=len(invalid_images), target_size=len(invalid_targets)), TemporaryDirectory() as folder:
                directory = Path(folder)
                self.write_cache(directory, invalid_images, invalid_targets)
                with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                    load_mnist(directory=directory)

    def test_labels_must_be_between_zero_and_nine(self):
        images, targets = self.make_idx()
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            for label in (10, 255):
                with self.subTest(label=label):
                    self.write_cache(directory, images, targets[:8] + bytes([0, label]))
                    with patch("examples.datasets.urlopen") as download:
                        with self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                            load_mnist(directory=directory)
                        download.assert_not_called()

    def test_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "train 或 test"):
            load_mnist("unknown")


if __name__ == "__main__":
    unittest.main()
