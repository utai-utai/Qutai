import random
import unittest
from contextlib import redirect_stdout
from io import StringIO
from math import isfinite, sqrt
from unittest.mock import Mock, patch

from qutai import Stream, Tensor, generation, nn


class StreamTests(unittest.TestCase):
    def test_ordered_stream_restarts_and_counts_partial_batch(self):
        data = Stream([[value] for value in range(5)], tuple(range(5)), batch_size=2, shuffle=False)
        self.assertEqual(len(data), 3)
        for _ in range(2):
            result = list(data)
            self.assertEqual([targets for _, targets in result], [[0, 1], [2, 3], [4]])
            self.assertEqual([inputs.shape[0] for inputs, _ in result], [2, 2, 1])

    def test_shuffle_changes_each_epoch_and_reproduces_the_whole_run(self):
        state = random.getstate()
        first = Stream([[value] for value in range(20)], tuple(range(20)), limit=12, seed=42)
        second = Stream([[value] for value in range(20)], tuple(range(20)), limit=12, seed=42)
        previous = None
        for _ in range(3):
            order = [target for _, targets in first for target in targets]
            self.assertEqual(order, [target for _, targets in second for target in targets])
            self.assertEqual(sorted(order), list(range(12)))
            self.assertNotEqual(order, previous)
            previous = order
        self.assertEqual(random.getstate(), state)

    def test_text_uses_the_same_stream_and_limits_complete_windows(self):
        data = Stream(list(range(20)), context_length=3, batch_size=2, limit=3, shuffle=False)
        self.assertEqual(len(data), 2)
        for _ in range(2):
            self.assertEqual(list(data), [
                ([[0, 1, 2], [3, 4, 5]], [[1, 2, 3], [4, 5, 6]]),
                ([[6, 7, 8]], [[7, 8, 9]]),
            ])

    def test_progress_counts_completed_batches_and_can_be_repeated(self):
        data = Stream([[value] for value in range(5)], tuple(range(5)), batch_size=2, progress=True)
        output = StringIO()
        with redirect_stdout(output):
            for _ in range(2):
                for inputs, targets in data:
                    self.assertEqual(inputs.shape[0], len(targets))
        self.assertEqual(output.getvalue().count("100.0%"), 2)
        self.assertEqual(output.getvalue().count("3/3"), 2)
        self.assertEqual(output.getvalue().count("\n"), 2)

    def test_failed_or_closed_iteration_does_not_report_completion(self):
        data = Stream([[1], [2]], (0, 1), batch_size=1, progress=True)
        output = StringIO()
        with redirect_stdout(output):
            iterator = iter(data)
            next(iterator)
            self.assertNotIn("100.0%", output.getvalue())
            iterator.close()
        self.assertNotIn("100.0%", output.getvalue())
        output = StringIO()
        with redirect_stdout(output), patch.object(Stream, "_batch", side_effect=RuntimeError("停止")):
            with self.assertRaises(RuntimeError): list(data)
        self.assertNotIn("100.0%", output.getvalue())

    def test_empty_stream_and_disabled_progress_are_quiet(self):
        output = StringIO()
        with redirect_stdout(output):
            empty = Stream(b"", b"", progress=True)
            self.assertEqual(len(empty), 0)
            self.assertEqual(list(empty), [])
            self.assertEqual(list(Stream([], progress=True)), [])
            list(Stream([[1]], (0,), progress=False))
        self.assertEqual(output.getvalue(), "")

    def test_tensor_shapes_are_inferred_without_image_assumptions(self):
        images = [Tensor([[[1, 2], [3, 4]]]), Tensor([[[5, 6], [7, 8]]])]
        data = Stream(images, [0, 1], shuffle=False, progress=False)
        inputs, targets = next(iter(data))
        self.assertEqual(inputs.shape, (2, 1, 2, 2))
        self.assertEqual(inputs.values, list(range(1, 9)))
        self.assertEqual(targets, [0, 1])

    def test_lists_and_scalars_are_batched_without_rescaling(self):
        inputs, _ = next(iter(Stream([[0, 255], [128, 64]], [0, 1], shuffle=False, progress=False)))
        self.assertEqual(inputs.shape, (2, 2))
        self.assertEqual(inputs.tolist(), [[0, 255], [128, 64]])
        inputs, _ = next(iter(Stream([2, 3], [0, 1], shuffle=False, progress=False)))
        self.assertEqual(inputs.shape, (2,))
        self.assertEqual(inputs.tolist(), [2, 3])

    def test_same_size_but_different_shapes_are_rejected_in_a_batch(self):
        data = Stream([Tensor([1, 2]), Tensor([[3, 4]])], [0, 1], shuffle=False, progress=False)
        with self.assertRaisesRegex(ValueError, "同一批样本的形状必须一致"):
            list(data)

    def test_each_batch_uses_its_own_first_sample_shape(self):
        samples = [Tensor([1, 2]), Tensor([3, 4]), Tensor([[[5, 6]]]), Tensor([[[7, 8]]])]
        batches = list(Stream(samples, [0, 1, 2, 3], batch_size=2, shuffle=False, progress=False))
        self.assertEqual([inputs.shape for inputs, _ in batches], [(2, 2), (2, 1, 1, 2)])

    def test_progress_stream_can_train_for_multiple_epochs(self):
        data = Stream([[0.0], [1.0], [0.0], [1.0]], (0, 1, 0, 1), batch_size=2, progress=True, seed=42)
        model = nn.Model([nn.Flatten(), nn.Linear(1, 2, seed=42)])
        optimizer = nn.SGD(model, learning_rate=0.1)
        loss_function = nn.CrossEntropy()
        before = [parameter.values.copy() for parameter in model.parameters()]
        with redirect_stdout(StringIO()):
            for _ in range(2):
                for inputs, targets in data:
                    optimizer.zero_grad()
                    loss = loss_function(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
        self.assertNotEqual(before, [parameter.values for parameter in model.parameters()])

    def test_native_generation_normalizes_images_and_uses_stream_random(self):
        samples = [Tensor([0], shape=(1, 1, 1)), Tensor([1], shape=(1, 1, 1))]
        for method in (generation.Diffusion(2), generation.FlowMatching(2)):
            with self.subTest(method=type(method).__name__):
                random = Mock()
                random.random.side_effect = [0, 1]
                random.randint.side_effect = [1, 2]
                random.gauss.return_value = 0.25
                with patch('qutai.core.data.Random', return_value=random):
                    inputs, times, targets = next(iter(Stream(samples, method=method, batch_size=2, shuffle=False, progress=False)))
                self.assertEqual(inputs.shape, (2, 1, 1, 1))
                self.assertEqual(targets.shape, inputs.shape)
                if isinstance(method, generation.FlowMatching):
                    self.assertEqual(times, [0, 1])
                    self.assertEqual(inputs.values, [0.25, 1])
                    self.assertEqual(targets.values, [-1.25, 0.75])
                else:
                    self.assertEqual(times, [0.5, 1])
                    self.assertEqual(targets.values, [0.25, 0.25])
                    for value, clean, alpha in zip(inputs.values, (-1, 1), method.alpha_bars[1:], strict=True):
                        self.assertAlmostEqual(value, sqrt(alpha) * clean + sqrt(1 - alpha) * 0.25)
        self.assertEqual([image.item() for image in samples], [0, 1])

    def test_generation_prepares_every_batch_and_keeps_partial_batch_each_epoch(self):
        samples = [Tensor([value / 4] * 4, shape=(1, 2, 2)) for value in range(5)]
        before = [sample.values[:] for sample in samples]
        for method in (generation.Diffusion(2), generation.FlowMatching(2)):
            with self.subTest(method=type(method).__name__), patch.object(method, 'prepare', wraps=method.prepare) as prepare:
                data = Stream(samples, method=method, batch_size=2, shuffle=False, progress=False, seed=42)
                self.assertEqual(len(data), 3)
                for _ in range(2):
                    batches = list(data)
                    self.assertEqual([inputs.shape[0] for inputs, _, _ in batches], [2, 2, 1])
                    for inputs, times, targets in batches:
                        self.assertEqual(inputs.shape, targets.shape)
                        self.assertEqual(len(times), inputs.shape[0])
                        self.assertTrue(all(0 <= time <= 1 for time in times))
                        self.assertTrue(all(isfinite(value) for value in (*inputs.values, *targets.values)))
                        self.assertFalse(inputs.requires_grad or targets.requires_grad)
                self.assertEqual(prepare.call_count, 6)
        self.assertEqual(before, [sample.values for sample in samples])

    def test_generation_seed_reproduces_whole_run_without_repeating_each_batch(self):
        samples = [Tensor([value / 4], shape=(1, 1, 1)) for value in range(5)]
        global_state = random.getstate()
        for method in (generation.Diffusion(2), generation.FlowMatching(2)):
            with self.subTest(method=type(method).__name__):
                first = Stream(samples, method=method, batch_size=2, progress=False, seed=42)
                second = Stream(samples, method=method, batch_size=2, progress=False, seed=42)
                previous = None
                for _ in range(2):
                    left = [(inputs.values[:], times[:], targets.values[:]) for inputs, times, targets in first]
                    right = [(inputs.values[:], times[:], targets.values[:]) for inputs, times, targets in second]
                    self.assertEqual(left, right)
                    self.assertNotEqual(left, previous)
                    previous = left
        self.assertEqual(random.getstate(), global_state)

    def test_generation_empty_data_and_explicit_labels(self):
        method = generation.FlowMatching(2)
        with patch.object(method, 'prepare', wraps=method.prepare) as prepare:
            self.assertEqual(list(Stream([], method=method, progress=False)), [])
            self.assertEqual(list(Stream([[1]], limit=0, method=method, progress=False)), [])
            prepare.assert_not_called()
        with self.assertRaisesRegex(ValueError, "不需要类别标签"):
            Stream([[1]], [0], method=method)


if __name__ == "__main__":
    unittest.main()
