"""卷积和最大池化的手算结果、任意输出梯度与预测模式。"""

import unittest
from math import prod
from random import Random

import qutai as qt
from qutai import Tensor, nn


class ConvolutionTests(unittest.TestCase):
    def check_gradient(self, operation, parents, seed):
        """中心差分验证各输入，输出梯度显式给出，避免只检查求和梯度。"""
        output = operation(*parents)
        output.backward(Tensor(seed, shape=output.shape))
        for parent_index, parent in enumerate(parents):
            self.assertIsNotNone(parent.grad)
            self.assertEqual(parent.grad.shape, parent.shape)
            for position, analytic in enumerate(parent.grad.values):
                high = [Tensor(tensor.storage.copy(), shape=tensor.shape) for tensor in parents]
                low = [Tensor(tensor.storage.copy(), shape=tensor.shape) for tensor in parents]
                high[parent_index].values[position] += 1e-6
                low[parent_index].values[position] -= 1e-6
                upper = sum(value * weight for value, weight in zip(operation(*high).values, seed, strict=True))
                lower = sum(value * weight for value, weight in zip(operation(*low).values, seed, strict=True))
                with self.subTest(parent=parent_index, position=position):
                    self.assertAlmostEqual(analytic, (upper - lower) / 2e-6, places=7)

    def test_conv2d_hand_calculation_and_plain_output(self):
        data = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        weight = [[[[1, 2], [0, -1]]]]
        self.assertEqual(qt.conv2d(data, weight), [[[[0, 2], [6, 8]]]])
        result = qt.conv2d(data, weight, [0.5])
        self.assertEqual(result, [[[[0.5, 2.5], [6.5, 8.5]]]])
        self.assertIs(type(result[0][0][0][0]), float)
        self.assertEqual(qt.conv2d([[[[1, 2], [3, 4]]]], [[[[1, 1], [1, 1]]]],
                                     stride=2, padding=1), [[[[1, 2], [3, 4]]]])
        self.assertEqual(data, [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])

    def test_conv2d_batches_channels_rectangular_stride_and_padding(self):
        data = [
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[10, 20, 30, 40], [50, 60, 70, 80]]],
            [[[-1, -2, -3, -4], [-5, -6, -7, -8]], [[7, 8, 9, 10], [11, 12, 13, 14]]],
        ]
        weight = [[[[1, 2]], [[1, 0]]], [[[0, -1]], [[0, 1]]]]
        output = qt.conv2d(Tensor(data), weight, [1, -2], stride=(1, 2), padding=(1, 0))
        self.assertEqual(output.shape, (2, 2, 4, 2))
        self.assertEqual(output.tolist(), [
            [[[1, 1], [16, 42], [68, 94], [1, 1]], [[-2, -2], [16, 34], [52, 70], [-2, -2]]],
            [[[1, 1], [3, -1], [-5, -9], [1, 1]], [[-2, -2], [8, 12], [16, 20], [-2, -2]]],
        ])

    def test_conv2d_tensor_result_for_any_tensor_argument(self):
        arguments = ([[[[2]]]], [[[[3]]]], [4])
        for index in range(3):
            with self.subTest(tensor_argument=index):
                values = list(arguments)
                values[index] = Tensor(values[index], requires_grad=True)
                result = qt.conv2d(*values)
                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.tolist(), [[[[10]]]])
                result.backward([[[[2]]]])
                self.assertEqual(values[index].grad.item(), (6, 4, 2)[index])

        data = Tensor([[[[1, 2], [3, 4]]]], requires_grad=True)
        weight = Tensor([[[[3]]]], requires_grad=True)
        qt.sum(qt.conv2d(data, weight)).backward()
        self.assertEqual(data.grad.tolist(), [[[[3, 3], [3, 3]]]])
        self.assertEqual(weight.grad.tolist(), [[[[10]]]])

    def test_conv2d_weighted_input_weight_and_bias_gradients(self):
        random = Random(120)
        parents = [Tensor([random.uniform(-0.8, 0.8) for _ in range(prod(shape))],
                          shape=shape, requires_grad=True)
                   for shape in ((2, 2, 2, 3), (2, 2, 2, 2), (2,))]
        seed = [0.2, -0.7, 1.3, 0.6, -0.3, 0.8, 0.4, -0.5, 1.1, 0.9, -0.2, 0.1]
        self.check_gradient(lambda *values: qt.conv2d(*values, stride=(1, 2), padding=(1, 0)),
                            parents, seed)

    def test_pooling_batches_channels_rectangular_stride_and_padding(self):
        data = [
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[-1, -2, -3, -4], [-5, -6, -7, -8]]],
            [[[8, 7, 6, 5], [4, 3, 2, 1]], [[-8, -7, -6, -5], [-4, -3, -2, -1]]],
        ]
        result = qt.max_pool2d(data, (2, 3), stride=(1, 2), padding=(1, 1))
        self.assertEqual(result, [
            [[[2, 4], [6, 8], [6, 8]], [[-1, -2], [-1, -2], [-5, -6]]],
            [[[8, 7], [8, 7], [4, 3]], [[-7, -5], [-3, -1], [-3, -1]]],
        ])
        self.assertIs(type(result[0][0][0][0]), float)
        tensor_result = qt.max_pool2d(Tensor(data), (2, 3), stride=(1, 2), padding=(1, 1))
        self.assertEqual(tensor_result.shape, (2, 2, 3, 2))
        self.assertEqual(tensor_result.tolist(), result)

    def test_pooling_default_stride_equals_kernel_size(self):
        data = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
        self.assertEqual(qt.max_pool2d(data, 2), [[[[6, 8], [14, 16]]]])

    def test_pooling_padding_does_not_replace_negative_values(self):
        tensor = Tensor([[[[-4, -3], [-2, -1]]]], requires_grad=True)
        output = qt.max_pool2d(tensor, 2, stride=1, padding=1)
        self.assertEqual(output.tolist(), [[[[-4, -3, -3], [-2, -1, -1], [-2, -1, -1]]]])
        qt.sum(output).backward()
        self.assertEqual(tensor.grad.tolist(), [[[[1, 2], [2, 4]]]])

    def test_pooling_first_tie_and_overlapping_gradients(self):
        tied = Tensor([[[[5, 5], [5, 5]]]], requires_grad=True)
        qt.max_pool2d(tied, 2).backward([[[[7]]]])
        self.assertEqual(tied.grad.tolist(), [[[[7, 0], [0, 0]]]])

        tensor = Tensor([[[[1, 2, 3], [4, 9, 6], [7, 8, 5]]]], requires_grad=True)
        qt.max_pool2d(tensor, 2, stride=1).backward([[[[1, 2], [3, 4]]]])
        self.assertEqual(tensor.grad.tolist(), [[[[0, 0, 0], [0, 10, 0], [0, 0, 0]]]])

    def test_pooling_weighted_gradient_without_ties(self):
        random = Random(17)
        tensor = Tensor([value / 10 for value in random.sample(range(-30, 30), 24)],
                        shape=(2, 2, 2, 3), requires_grad=True)
        seed = [random.uniform(-1, 1) for _ in range(24)]
        self.check_gradient(lambda data: qt.max_pool2d(data, (2, 2), stride=(1, 1), padding=(1, 0)),
                            (tensor,), seed)

    def test_model_prediction_does_not_record_graph(self):
        layer = nn.Conv2d(1, 2, (2, 3), stride=(1, 2), padding=(0, 1), seed=42)
        model = nn.Model([layer, nn.MaxPool2d(2)])
        inputs = Tensor([value / 10 for value in range(32)], shape=(1, 1, 4, 8), requires_grad=True)
        training = model(inputs)
        self.assertTrue(training.requires_grad)
        prediction = model(inputs, train=False)
        self.assertEqual(prediction.tolist(), training.tolist())
        self.assertFalse(prediction.requires_grad)
        self.assertEqual(prediction.parents, ())
        self.assertIsNone(prediction.backward_rule)
        self.assertIsNone(inputs.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        qt.sum(model(inputs)).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_invalid_shapes_and_dimensions(self):
        data, weight = [[[[1, 2], [3, 4]]]], [[[[1]]]]
        for arguments in (([[1, 2], [3, 4]], weight), (data, [[[[1]], [[2]]]])):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    qt.conv2d(*arguments)
        with self.assertRaises(ValueError):
            qt.conv2d(data, weight, stride=0)
        with self.assertRaises(ValueError):
            qt.max_pool2d(data, 0)
        with self.assertRaises(ValueError):
            qt.max_pool2d(data, 2, padding=-1)


if __name__ == "__main__":
    unittest.main()
