"""像素 DiT 的 patch 复原、时间条件和非零调制下的数值梯度。"""

import math
import random
import unittest

from qutai import Tensor, nn
from qutai.core.context import is_recording, is_training, set_recording
from qutai.nn.dit import DiTBlock, TimeEmbedding


def activate(model):
    """打开初始为零的门控和输出，让梯度检查经过完整网络。"""
    model.output.weight.values = [(index % 5 - 2) / 5 for index in range(model.output.weight.size)]
    model.output_norm.eps = 0.2
    for block in model.blocks:
        block.attention_norm.eps = block.feed_forward_norm.eps = 0.2
        projection = block.modulation.layers[-1]
        projection.weight.values = [(index % 7 - 3) / 30 for index in range(projection.weight.size)]
        projection.bias.values = [0.1] * projection.bias.size


class DiTTests(unittest.TestCase):
    def test_zero_output_and_identity_blocks_are_initialization_choices(self):
        model = nn.DiT(image_size=(1, 2), patch_size=1, features=2, heads=1, hidden_features=3, seed=7)
        inputs = Tensor([0.2, -0.4], shape=(1, 1, 1, 2), requires_grad=True)
        result = model(inputs, [0.3])
        self.assertEqual(result.shape, inputs.shape)
        self.assertEqual(result.values, [0, 0])
        result.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertEqual(model.output.bias.grad.tolist(), [2])
        block = DiTBlock(2, 1, hidden_features=3, seed=7)
        tokens = Tensor([[[0.2, -0.4], [0.7, 0.1]]])
        self.assertEqual(block(tokens, [[1, 2]]).tolist(), tokens.tolist())

    def test_non_square_multichannel_patch_projection_can_reconstruct_exact_pixels(self):
        model = nn.DiT(2, (2, 4), patch_size=(1, 2), features=4, heads=2, seed=7)
        convolution = model.patch_embedding.projection
        convolution.weight.values = [float(row == column) for row in range(4) for column in range(4)]
        convolution.bias.values = [0.0] * 4
        model.position_embedding.weight.values = [0.0] * model.position_embedding.weight.size
        model.output_norm = nn.Identity()
        model.output.weight.values = [float(row == column) for row in range(4) for column in range(4)]
        inputs = Tensor(list(range(32)), shape=(2, 2, 2, 4), requires_grad=True)
        result = model(inputs, [0.0, 1.0])
        self.assertEqual(result.tolist(), inputs.tolist())
        result.sum().backward()
        self.assertEqual(inputs.grad.values, [1.0] * 32)

    def test_all_small_network_parameters_and_input_match_central_differences(self):
        model = nn.DiT(image_size=(1, 2), patch_size=1, features=2, heads=1, hidden_features=3, seed=7)
        activate(model)
        inputs = Tensor([0.2, -0.4], shape=(1, 1, 1, 2), requires_grad=True)
        result = model(inputs, [0.3])
        seed = [0.3, -0.7]
        result.backward(Tensor(seed, shape=result.shape))
        for parameter in (inputs, *model.parameters()):
            self.assertIsNotNone(parameter.grad)
            for index, analytic in enumerate(parameter.grad.values):
                with self.subTest(shape=parameter.shape, index=index):
                    original = parameter.values[index]
                    parameter.values[index] = original + 1e-6
                    upper = model(inputs, [0.3], train=False)
                    parameter.values[index] = original - 1e-6
                    lower = model(inputs, [0.3], train=False)
                    parameter.values[index] = original
                    numeric = math.fsum((a - b) * g for a, b, g in zip(upper.values, lower.values, seed, strict=True)) / 2e-6
                    self.assertAlmostEqual(analytic, numeric, places=6)
        self.assertTrue(any(abs(value) > 1e-12 for value in inputs.grad.values))
        self.assertTrue(any(abs(value) > 1e-12 for parameter in model.time_embedding.parameters() for value in parameter.grad.values))

    def test_time_condition_changes_output_independently_for_each_sample(self):
        model = nn.DiT(image_size=(1, 2), patch_size=1, features=2, heads=1, hidden_features=3, seed=7)
        activate(model)
        single = Tensor([0.2, -0.4], shape=(1, 1, 1, 2))
        batch = Tensor([0.2, -0.4, 0.2, -0.4], shape=(2, 1, 1, 2))
        result = model(batch, [0.2, 0.8], train=False)
        self.assertEqual(result.values[:2], model(single, [0.2], train=False).values)
        self.assertEqual(result.values[2:], model(single, [0.8], train=False).values)
        self.assertNotEqual(result.values[:2], result.values[2:])

    def test_time_encoding_uses_continuous_constants_and_supports_odd_width(self):
        encoding = TimeEmbedding(4, seed=7)
        encoding.layers = ()
        result = encoding([0.0, 0.25])
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(result.values[:4], [1, 1, 0, 0])
        for actual, expected in zip(result.values[4:], [math.cos(250), math.cos(0.025), math.sin(250), math.sin(0.025)], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(result.requires_grad)
        self.assertEqual(TimeEmbedding(3, seed=7)([0.5]).shape, (1, 3))

    def test_eval_and_outer_disabled_recording_preserve_context(self):
        model = nn.DiT(image_size=2, patch_size=1, features=2, heads=1, seed=7)
        inputs = Tensor([1, 2, 3, 4], shape=(1, 1, 2, 2), requires_grad=True)
        recording, training = is_recording(), is_training()
        prediction = model(inputs, [0.5], train=False)
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertEqual((is_recording(), is_training()), (recording, training))
        with set_recording(False):
            self.assertFalse(model(inputs, [0.5]).requires_grad)
            self.assertFalse(is_recording())
        self.assertTrue(model(inputs, [0.5]).requires_grad)
        self.assertTrue(inputs.requires_grad)

    def test_empty_batch_returns_same_shape_and_zero_gradients(self):
        model = nn.DiT(image_size=2, patch_size=1, features=2, heads=1, seed=7)
        inputs = Tensor([], shape=(0, 1, 2, 2), requires_grad=True)
        result = model(inputs, [])
        self.assertEqual(result.shape, inputs.shape)
        result.sum().backward()
        self.assertEqual(inputs.grad.values, [])
        self.assertTrue(all(parameter.grad.values == [0.0] * parameter.size for parameter in model.parameters()))

    def test_seed_is_local_and_reproducible(self):
        state = random.getstate()
        first = nn.DiT(image_size=2, patch_size=1, features=2, heads=1, seed=7)
        second = nn.DiT(image_size=2, patch_size=1, features=2, heads=1, seed=7)
        self.assertEqual(random.getstate(), state)
        self.assertEqual([p.values for p in first.parameters()], [p.values for p in second.parameters()])

    def test_invalid_images_times_and_configuration_raise(self):
        model = nn.DiT(image_size=2, patch_size=1, features=2, heads=1, seed=7)
        inputs = Tensor([1, 2, 3, 4], shape=(1, 1, 2, 2))
        for times in ([], [0.1, 0.2], [-0.1], [1.1], [float("nan")], [float("inf")], [True]):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "[\u4e00-\u9fff]"):
                model(inputs, times)
        for images in ([[1, 2]], Tensor([], shape=(0, 2, 2, 2))):
            with self.subTest(images=images), self.assertRaises(ValueError):
                model(images, [])
        for arguments in ({"image_size": 0}, {"image_size": 3, "patch_size": 2}, {"depth": 0}, {"features": 3, "heads": 2}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                nn.DiT(**arguments)


if __name__ == "__main__":
    unittest.main()
