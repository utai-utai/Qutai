import re
import unittest
from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree

from qutai import Tensor, generation, nn
from qutai.visualization import Image, ImageCombine, Network


class GenerationVisualizationTests(unittest.TestCase):
    def test_tensor_display_clips_pixels_without_mutating_values_or_gradient(self):
        tensor = Tensor([-2, -1, 0, 1, 2], shape=(1, 1, 5), requires_grad=True)
        gradient = Tensor([3] * 5, shape=tensor.shape)
        tensor.grad = gradient
        before = tensor.values[:]
        image = Image.from_tensor(tensor, scale=1)
        self.assertEqual(image.pixels, bytes([0, 0, 128, 255, 255]))
        self.assertEqual(image.shape, (1, 1, 5))
        self.assertEqual(image.scale, 1)
        self.assertEqual(tensor.values, before)
        self.assertEqual(tensor.shape, (1, 1, 5))
        self.assertTrue(tensor.requires_grad)
        self.assertIs(tensor.grad, gradient)
        self.assertEqual(tensor.parents, ())

    def test_custom_tensor_range_preserves_chw_rgb_layout(self):
        tensor = Tensor([0, 1, 0.5, 0.25, 1, 0], shape=(3, 1, 2))
        image = Image.from_tensor(tensor, value_range=(0, 1), scale=2)
        expected = Image(bytes([0, 255, 128, 64, 255, 0]), (3, 1, 2), scale=2)
        self.assertEqual(image._repr_png_(), expected._repr_png_())

    def test_invalid_tensor_rank_and_display_ranges(self):
        for tensor in (Tensor([0]), Tensor([[0]]), Tensor([0], shape=(1, 1, 1, 1))):
            with self.subTest(shape=tensor.shape):
                with self.assertRaisesRegex(ValueError, "三维"):
                    Image.from_tensor(tensor)
        for limits in ((1, 1), (2, -1)):
            with self.subTest(limits=limits):
                with self.assertRaisesRegex(ValueError, "上限"):
                    Image.from_tensor(Tensor([0], shape=(1, 1, 1)), value_range=limits)

    def test_grid_embeds_actual_pngs_escapes_labels_and_accepts_generators(self):
        images = [Image(bytes([value]), (1, 1, 1), scale=1) for value in (0, 255)]
        grid = ImageCombine((image for image in images), columns=2, labels=['<script>&"', '第二张'], title='<轨迹>')
        html = grid._repr_html_()
        self.assertIn('grid-template-columns:repeat(2,max-content)', html)
        self.assertIn('&lt;script&gt;&amp;&quot;', html)
        self.assertIn('&lt;轨迹&gt;', html)
        self.assertNotIn('<script>', html)
        sources = re.findall(r'src="([^"]+)"', html)
        self.assertEqual(len(sources), 2)
        for source, image in zip(sources, images, strict=True):
            self.assertTrue(source.startswith('data:image/png;base64,'))
            self.assertEqual(b64decode(source.split(',', 1)[1]), image._repr_png_())
        self.assertNotIn('src="http', html)

    def test_grid_save_is_standalone_html_and_zero_label_is_visible(self):
        grid = ImageCombine([Image(bytes([128]), (1, 1, 1), scale=1)], labels=[0], title='生成过程')
        html = grid._repr_html_()
        self.assertIn('>0</figcaption>', html)
        with TemporaryDirectory() as folder:
            path = grid.save(Path(folder) / 'trajectory.html')
            saved = path.read_text(encoding='utf-8')
        self.assertTrue(saved.startswith('<!doctype html>'))
        self.assertIn(html, saved)
        self.assertIn('data:image/png;base64,', saved)
        self.assertIn('暂无图像', ImageCombine([])._repr_html_())

    def test_grid_rejects_invalid_columns_and_label_counts(self):
        image = Image(bytes([0]), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "列数"):
            ImageCombine([image], columns=0)
        with self.assertRaisesRegex(ValueError, "标签数量"):
            ImageCombine([image], labels=[])

    def test_dit_network_keeps_image_and_time_inputs_separate_without_forward(self):
        network = nn.DiT(1, 4, patch_size=2, features=4, heads=2, depth=1, hidden_features=8, seed=42)
        model = generation.Model(network, method=generation.Diffusion(steps=8))
        before = [parameter.values[:] for parameter in network.parameters()]
        with patch.object(nn.DiT, '__call__', side_effect=AssertionError('不应执行 DiT')), patch.object(generation.Model, '__call__', side_effect=AssertionError('不应执行生成模型')), patch.object(generation.Model, 'sample', side_effect=AssertionError('不应采样')):
            direct = Network(network, title='DiT')
            wrapped = Network(model, title='Diffusion')
            ElementTree.fromstring(direct._repr_svg_())
            ElementTree.fromstring(wrapped._repr_svg_())
        self.assertEqual(direct.nodes, wrapped.nodes)
        self.assertEqual(direct.edges, wrapped.edges)
        incoming = {edge[1] for edge in direct.edges}
        sources = {label for node, label in direct.nodes.items() if node not in incoming}
        self.assertEqual(sources, {'Image\n1×4×4', 'Time t'})
        time = next(node for node, label in direct.nodes.items() if label.startswith('TimeEmbedding\n'))
        block = next(node for node, label in direct.nodes.items() if label.startswith('DiTBlock\n'))
        self.assertIn((time, block, 'time'), direct.edges)
        self.assertIn('Unpatchify\n1×4×4', direct.nodes.values())
        self.assertEqual(before, [parameter.values for parameter in network.parameters()])
        self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))


if __name__ == '__main__':
    unittest.main()
