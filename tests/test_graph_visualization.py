"""无向图、MaxCut 分组与独立 SVG 保存。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from qutai.visualization import Graph


NS = {'svg': 'http://www.w3.org/2000/svg'}


class GraphVisualizationTests(unittest.TestCase):
    def test_cycle_uses_undirected_edges_and_accepts_generators(self):
        graph = Graph(iter(range(4)), iter([(0, 1), (1, 2), (2, 3), (3, 0)]), title='四个节点的环')
        drawing = graph._repr_svg_()
        root = ElementTree.fromstring(drawing)
        self.assertEqual(len(root.findall('.//svg:line', NS)), 4)
        self.assertEqual(len(root.findall('.//svg:g[@data-node]', NS)), 4)
        self.assertNotIn('marker-end', drawing)
        self.assertEqual(root.findall('.//svg:line', NS)[-1].get('data-target'), '0')

    def test_partition_colors_nodes_and_highlights_only_cross_group_edges(self):
        graph = Graph(range(4), [(0, 1), (1, 2), (2, 3), (3, 0)], groups={0: 0, 1: 0, 2: 1, 3: 1})
        root = ElementTree.fromstring(graph._repr_svg_())
        edges = root.findall('.//svg:line', NS)
        self.assertEqual([edge.get('data-cut') for edge in edges], ['false', 'true', 'false', 'true'])
        self.assertEqual(len({edge.get('stroke') for edge in edges}), 2)
        fills = [group.find('svg:circle', NS).get('fill') for group in root.findall('.//svg:g', NS)]
        self.assertEqual(fills[0], fills[1])
        self.assertEqual(fills[2], fills[3])
        self.assertNotEqual(fills[0], fills[2])
        self.assertIn('跨组边：2 / 4', ''.join(root.itertext()))

    def test_partial_groups_do_not_assume_an_unassigned_node_is_in_group_zero(self):
        graph = Graph(range(3), [(0, 1), (1, 2), (2, 0)], groups={0: 0, 1: 1})
        root = ElementTree.fromstring(graph._repr_svg_())
        self.assertEqual([edge.get('data-cut') for edge in root.findall('.//svg:line', NS)], ['true', 'false', 'false'])
        self.assertIn('跨组边：1 / 3', ''.join(root.itertext()))

    def test_title_and_node_ids_are_escaped_as_text_and_attributes(self):
        node = '<node>&"'
        graph = Graph([node, '另外一个'], [(node, '另外一个')], title='<图>&"')
        drawing = graph._repr_svg_()
        root = ElementTree.fromstring(drawing)
        self.assertEqual(root.find('svg:title', NS).text, '<图>&"')
        self.assertEqual(root.find('.//svg:line', NS).get('data-source'), node)
        self.assertIn(node, ''.join(root.itertext()))
        self.assertNotIn('<node>', drawing)

    def test_snapshot_and_saved_svg_preserve_original_inputs(self):
        nodes, edges, groups = [0, 1], [[0, 1]], {0: 0, 1: 1}
        graph = Graph(nodes, edges, groups=groups, title='快照')
        before = graph._repr_svg_()
        nodes.append(2)
        edges[0][1] = 2
        groups[1] = 0
        self.assertEqual(graph._repr_svg_(), before)
        with TemporaryDirectory() as directory:
            path = graph.save(Path(directory) / 'graph.svg')
            self.assertEqual(path.read_text(encoding='utf-8'), before)
        self.assertEqual(nodes, [0, 1, 2])
        self.assertEqual(edges, [[0, 2]])
        self.assertEqual(groups, {0: 0, 1: 0})

    def test_empty_graph_and_isolated_node_are_displayable(self):
        empty = ElementTree.fromstring(Graph([], [])._repr_svg_())
        self.assertIn('暂无节点', ''.join(empty.itertext()))
        self.assertEqual(empty.findall('.//svg:g', NS), [])
        single = ElementTree.fromstring(Graph([0], [])._repr_svg_())
        self.assertEqual(len(single.findall('.//svg:g', NS)), 1)
        self.assertEqual(single.findall('.//svg:line', NS), [])

    def test_unknown_edge_endpoint_or_group_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '已有节点'): Graph([0], [(0, 1)])
        with self.assertRaisesRegex(ValueError, '已有节点'): Graph([0], [], groups={1: 0})


if __name__ == '__main__':
    unittest.main()
