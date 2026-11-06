"""Iris Notebook 的数据划分、标准化、训练收敛和留出集预测。"""

import unittest
from collections import Counter
from statistics import mean, pstdev

import qutai as qt
from tests.notebook_helpers import run_notebook


class ClassificationTrainingTests(unittest.TestCase):
    def test_classifier_learns_and_generalizes(self):
        experiment = run_notebook("classification")
        model = experiment["model"]
        loss = experiment["loss_function"](model(experiment["inputs"], train=False), experiment["targets"])
        self.assertLess(loss.item(), 0.2)

        self.assertEqual(experiment["names"], ["setosa", "versicolor", "virginica"])
        self.assertEqual(len(experiment["inputs"]), 120)
        self.assertEqual(Counter(experiment["targets"]), {0: 40, 1: 40, 2: 40})
        self.assertTrue(all(len(sample) == 4 for sample in experiment["inputs"]))
        train_indices, test_indices = map(set, (experiment["train_indices"], experiment["test_indices"]))
        self.assertEqual(len(train_indices), 120)
        self.assertEqual(len(test_indices), 30)
        self.assertTrue(train_indices.isdisjoint(test_indices))
        self.assertEqual(train_indices | test_indices, set(range(150)))
        for column in zip(*experiment["inputs"]):
            self.assertAlmostEqual(mean(column), 0, places=12)
            self.assertAlmostEqual(pstdev(column), 1, places=12)

        held_out, expected = experiment["test_inputs"], experiment["test_targets"]
        self.assertEqual(len(held_out), 30)
        self.assertEqual(Counter(expected), {0: 10, 1: 10, 2: 10})
        scores = model(held_out, train=False)
        predictions = qt.argmax(scores, axis=-1)
        accuracy = sum(guess == target for guess, target in zip(predictions, expected)) / len(expected)
        self.assertGreaterEqual(accuracy, 0.9)
        self.assertEqual(scores.shape, (30, 3))
        self.assertFalse(scores.requires_grad)
        self.assertEqual(scores.parents, ())


if __name__ == "__main__":
    unittest.main()
