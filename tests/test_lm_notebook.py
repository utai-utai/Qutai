"""语言模型 Notebook 的两次 AdamW 更新、裁剪和采样输出。"""

import json
import unittest
from math import hypot, isfinite
from pathlib import Path
from unittest.mock import patch

from qutai import nn
from tests.notebook_helpers import run_notebook


class LanguageNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(nn.AdamW, "step", autospec=True, side_effect=nn.AdamW.step) as step:
            cls.namespace = run_notebook("language_model")
            cls.updates = step.call_count

    def test_exactly_two_updates_with_finite_decreasing_loss_and_clipped_gradients(self):
        self.assertEqual(self.updates, 2)
        history = self.namespace["history"]
        self.assertEqual([row[0] for row in history], [0, 1, 2])
        self.assertTrue(all(isfinite(row[1]) for row in history))
        self.assertLess(history[-1][1], history[0][1])
        model, optimizer = self.namespace["model"], self.namespace["optimizer"]
        norm = hypot(*(value for parameter in model.parameters() for value in parameter.grad.values))
        self.assertLessEqual(norm, 1 + 1e-12)
        self.assertTrue(all(optimizer.steps[parameter] == 2 for parameter in model.parameters()))

    def test_sampling_is_reproducible_and_notebook_keeps_visual_outputs(self):
        model = self.namespace["model"]
        first = model.generate("小猫", length=16, temperature=0.8, top_k=5, seed=42)
        self.assertEqual(first, model.generate("小猫", length=16, temperature=0.8, top_k=5, seed=42))
        self.assertTrue(first.startswith("小猫"))
        self.assertEqual(len(first), 18)
        path = Path(__file__).resolve().parents[1] / "examples/language_model.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn(".save(", code)
        svg_count = sum("image/svg+xml" in output.get("data", {})
                        for cell in notebook["cells"] for output in cell.get("outputs", []))
        self.assertEqual(svg_count, 2)

    def test_notebook_cache_comparison_reduces_projection_work_and_keeps_the_text(self):
        model = self.namespace["model"]
        self.assertEqual(self.namespace["without_cache"], self.namespace["with_cache"])
        projection = model.network.layers[1].attention.key
        for use_cache, expected in ((False, [2, 3, 4, 5, 6, 7]), (True, [2, 1, 1, 1, 1, 1])):
            with self.subTest(use_cache=use_cache):
                with patch.object(nn.Linear, "__call__", autospec=True, side_effect=nn.Linear.__call__) as project:
                    result = model.generate("小猫", length=6, use_cache=use_cache)
                lengths = [call.args[1].shape[1] for call in project.call_args_list if call.args[0] is projection]
                self.assertEqual(lengths, expected)
                self.assertEqual(result, self.namespace["with_cache"])


if __name__ == "__main__":
    unittest.main()
