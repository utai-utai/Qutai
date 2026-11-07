import csv
import unittest
from math import isfinite
from pathlib import Path
from struct import unpack_from
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree
from zlib import crc32, decompress

from qutai import lm, nn
from qutai.visualization import Diagram, Heatmap, Image, Network, Plot, Table


SVG = "{http://www.w3.org/2000/svg}"


class VisualizationTests(unittest.TestCase):
    def test_plot_chaining_negative_coordinates_axes_legend_and_escaping(self):
        plot = Plot('loss <train>', xlabel='step & epoch', ylabel='MSE')
        self.assertIs(plot.line([-3, -2, -1], [3, 1, 0], label='train & eval'), plot)
        self.assertIs(plot.scatter([-3, -1], [-2, -2], label='<samples>'), plot)
        content = plot._repr_svg_()
        root = ElementTree.fromstring(content)
        self.assertEqual(root.tag, SVG + 'svg')
        self.assertEqual(len(root.findall(SVG + 'polyline')), 1)
        self.assertEqual(len(root.findall(SVG + 'circle')), 2)
        self.assertIn('step & epoch', ''.join(root.itertext()))
        self.assertIn('train & eval', ''.join(root.itertext()))
        self.assertIn('&lt;samples&gt;', content)
        self.assertNotIn('<samples>', content)
        with TemporaryDirectory() as folder:
            path = plot.save(Path(folder) / 'loss.svg')
            self.assertEqual(path.read_text(), content)

    def test_empty_and_constant_plots_have_finite_coordinates(self):
        for plot in (Plot(), Plot().line([], []), Plot().line([2, 2], [-5, -5])):
            with self.subTest(series=plot.series):
                root = ElementTree.fromstring(plot._repr_svg_())
                for line in root.findall(SVG + 'polyline'):
                    self.assertTrue(all(isfinite(float(value)) for point in line.attrib['points'].split() for value in point.split(',')))
        with self.assertRaises(ValueError):
            Plot().line([1, 2], [1])

    def test_table_csv_limit_empty_data_and_html_escaping(self):
        table = Table([{'name': '<script>&"', 'value': 3}], title='<results>')
        html = table._repr_html_()
        self.assertIn('&lt;script&gt;&amp;&quot;', html)
        self.assertIn('&lt;results&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertIn('<th', html)
        self.assertIn('暂无数据', Table([], columns=['value'])._repr_html_())
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'data.csv'
            with source.open('w', newline='', encoding='utf-8') as file:
                csv.writer(file).writerows([['name', 'value'], ['a,b', '1'], ['c', '2']])
            loaded = Table.from_csv(source, limit=1)
            self.assertEqual(loaded.columns, ['name', 'value'])
            self.assertEqual(loaded.rows, [['a,b', '1']])
            self.assertEqual(Table.from_csv(source, limit=0).rows, [])
            saved = table.save(Path(folder) / 'table.html').read_text()
            self.assertTrue(saved.startswith('<!doctype html>'))
            self.assertIn(html, saved)

    def test_heatmap_values_labels_empty_and_constant_data(self):
        heatmap = Heatmap([[-1, 0], [0.5, 1]], xlabels=['<a>', 'b'], ylabels=['q&k', 'v'], title='Weights')
        root = ElementTree.fromstring(heatmap._repr_svg_())
        cells = root.findall(SVG + 'rect')[1:]
        self.assertEqual(len(cells), 4)
        self.assertEqual(len({cell.attrib['fill'] for cell in cells}), 4)
        self.assertIn('<a>', ''.join(root.itertext()))
        self.assertIn('q&k', ''.join(root.itertext()))
        for data in ([], [[]], [[0, 0], [0, 0]]):
            ElementTree.fromstring(Heatmap(data)._repr_svg_())
        with self.assertRaises(ValueError):
            Heatmap([[1], [2, 3]])

    def test_png_preserves_original_gray_rgb_and_scaled_scanlines(self):
        fixtures = (
            (bytes([0, 255, 128, 64]), (1, 2, 2), 4, 4, 0,
             b'\x00\x00\x00\xff\xff' * 2 + b'\x00\x80\x80\x40\x40' * 2),
            (bytes([255, 0, 0, 255, 0, 0]), (3, 1, 2), 4, 2, 2,
             (b'\x00' + bytes([255, 0, 0]) * 2 + bytes([0, 255, 0]) * 2) * 2),
        )
        for pixels, shape, width, height, color, scanlines in fixtures:
            with self.subTest(shape=shape), TemporaryDirectory() as folder:
                image = Image(pixels, shape, scale=2)
                content = image._repr_png_()
                # 压缩结果可能随 zlib 版本变化，核对 PNG 结构和解压后的像素。
                self.assertEqual(content[:8], b'\x89PNG\r\n\x1a\n')
                chunks, offset = {}, 8
                while offset < len(content):
                    length = unpack_from('>I', content, offset)[0]
                    kind = content[offset + 4:offset + 8]
                    payload = content[offset + 8:offset + 8 + length]
                    self.assertEqual(unpack_from('>I', content, offset + 8 + length)[0], crc32(kind + payload))
                    chunks[kind] = payload
                    offset += length + 12
                self.assertEqual(offset, len(content))
                self.assertEqual(list(chunks), [b'IHDR', b'IDAT', b'IEND'])
                self.assertEqual(chunks[b'IEND'], b'')
                self.assertEqual(unpack_from('>IIBBBBB', chunks[b'IHDR']), (width, height, 8, color, 0, 0, 0))
                self.assertEqual(decompress(chunks[b'IDAT']), scanlines)
                self.assertEqual(image.save(Path(folder) / 'image.png').read_bytes(), content)

    def test_diagram_generic_labels_branches_and_svg_saving(self):
        diagram = Diagram({'input': '<input>', 'q': 'Q', 'k': 'K', 'scores': 'QKᵀ'},
                          [('input', 'q'), ('input', 'k'), ('q', 'scores', '<query>'), ('k', 'scores')], title='Q & K')
        content = diagram._repr_svg_()
        root = ElementTree.fromstring(content)
        self.assertEqual(len(root.findall(SVG + 'g')), 4)
        self.assertIn('&lt;input&gt;', content)
        self.assertIn('&lt;query&gt;', content)
        for value in (Diagram([], []), Diagram(['first', 'second'], [('first', 'second')]), Heatmap([[1]])):
            with TemporaryDirectory() as folder:
                path = value.save(Path(folder) / 'diagram.svg')
                ElementTree.fromstring(path.read_text())

    def test_network_draws_residual_branches_without_running_model(self):
        branch = nn.Model([nn.Conv2d(1, 2, 1, seed=42), nn.BatchNorm2d(2)])
        shortcut = nn.Model([nn.Conv2d(1, 2, 1, seed=43), nn.BatchNorm2d(2)])
        model = nn.Model([branch + shortcut, nn.ReLU(), nn.GlobalAvgPool2d()])
        before = [(layer.running_mean[:], layer.running_var[:]) for layer in (branch.layers[1], shortcut.layers[1])]
        parameters = [parameter.values[:] for parameter in model.parameters()]
        with patch.object(nn.Model, '__call__', side_effect=AssertionError('不应执行模型')), patch.object(nn.BatchNorm2d, '__call__', side_effect=AssertionError('不应更新统计量')):
            network = Network(model, title='Residual network')
            ElementTree.fromstring(network._repr_svg_())
        add = next(node for node, label in network.nodes.items() if label == 'Add (+)')
        incoming = [edge for edge in network.edges if edge[1] == add]
        self.assertEqual({edge[2] for edge in incoming}, {'main', 'shortcut'})
        self.assertNotEqual(incoming[0][0], incoming[1][0])
        self.assertEqual(before, [(layer.running_mean, layer.running_var) for layer in (branch.layers[1], shortcut.layers[1])])
        self.assertEqual(parameters, [parameter.values for parameter in model.parameters()])
        identity = Network(nn.Residual(nn.Model(), nn.Model()))
        self.assertEqual(list(identity.nodes.values()).count('Identity'), 2)

    def test_network_collapses_resnet_and_labels_attention_dimensions(self):
        block = Network(nn.ResNetBlock(1, 2, stride=2, seed=100))
        self.assertEqual(len(block.nodes), 3)
        self.assertTrue(any(label.startswith('ResNetBlock\n1 → 2') for label in block.nodes.values()))
        attention = Network(nn.Attention(2, 8, seed=42))
        self.assertIn('Attention\n2 → 8', attention.nodes.values())
        with TemporaryDirectory() as folder:
            ElementTree.fromstring(attention.save(Path(folder) / 'network.svg').read_text())

    def test_network_labels_vit_layers_and_collapses_transformer_without_forward(self):
        model = nn.Model([
            nn.PatchEmbedding(1, 8, patch_size=7, seed=42),
            nn.PositionEmbedding(16, 8, seed=43),
            nn.TransformerBlock(8, heads=2, hidden_features=16, seed=44),
            nn.LayerNorm(8), nn.Mean(axis=1), nn.Linear(8, 10, seed=45),
        ])
        parameters = [parameter.values[:] for parameter in model.parameters()]
        with patch.object(nn.Model, '__call__', side_effect=AssertionError('不应执行模型')), patch.object(nn.TransformerBlock, '__call__', side_effect=AssertionError('不应执行模块')), patch.object(nn.MultiHeadAttention, '__call__', side_effect=AssertionError('不应计算注意力')):
            network = Network(model, title='Tiny ViT')
            ElementTree.fromstring(network._repr_svg_())
            attention = Network(nn.MultiHeadAttention(8, heads=2, seed=50))
            ElementTree.fromstring(attention._repr_svg_())
        self.assertEqual(len(network.nodes), 8)
        self.assertIn('PatchEmbedding\n1 → 8 · 7×7', network.nodes.values())
        self.assertIn('PositionEmbedding\n16 tokens · 8 features', network.nodes.values())
        self.assertIn('TransformerBlock\n8 features · 2 heads', network.nodes.values())
        self.assertIn('LayerNorm\n8 features', network.nodes.values())
        self.assertIn('Mean\naxis=1', network.nodes.values())
        self.assertIn('MultiHeadAttention\n8 features · 2 heads', attention.nodes.values())
        self.assertEqual(parameters, [parameter.values for parameter in model.parameters()])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_language_model_context_vocabulary_and_embedding_without_forward(self):
        tokenizer = lm.Tokenizer('小猫看雨。小狗看风。')
        model = lm.Model(tokenizer, context_length=8, features=8, heads=2, depth=1, hidden_features=16, seed=42)
        parameters = [parameter.values[:] for parameter in model.parameters()]
        with patch.object(lm.Model, '__call__', side_effect=AssertionError('不应执行语言模型')), patch.object(lm.Model, 'generate', side_effect=AssertionError('不应生成文字')), patch.object(nn.Embedding, '__call__', side_effect=AssertionError('不应查词表')):
            network = Network(model, title='字级语言模型')
            ElementTree.fromstring(network._repr_svg_())
        self.assertEqual(len(network.nodes), 7)
        self.assertIn('Character ids\ncontext ≤ 8', network.nodes.values())
        self.assertIn(f'Embedding\n{tokenizer.size} ids → 8 features', network.nodes.values())
        self.assertIn(f'Next character\n{tokenizer.size} scores/token', network.nodes.values())
        self.assertIn('TransformerBlock\n8 features · 2 heads', network.nodes.values())
        self.assertEqual(parameters, [parameter.values for parameter in model.parameters()])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(list(Network(lm.Model()).nodes.values()), ['Input', 'Output'])


if __name__ == '__main__':
    unittest.main()
