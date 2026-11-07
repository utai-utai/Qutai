import unittest

from qutai import Tensor, nn


class MSETests(unittest.TestCase):
    def setUp(self):
        self.loss_function = nn.MSE()

    def test_plain_data_returns_scalar_tensor(self):
        loss = self.loss_function((1, 2, 3), [0, 2, 5])
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(loss.item(), 5 / 3)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(self.loss_function(3, 1).item(), 4)

    def test_matrix_loss_and_both_gradients(self):
        prediction = Tensor([[1, 2], [3, 4]], requires_grad=True)
        target = Tensor([[0, 2], [1, 5]], requires_grad=True)
        loss = self.loss_function(prediction, target)
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertEqual(prediction.grad.tolist(), [[0.5, 0], [1, -0.5]])
        self.assertEqual(target.grad.tolist(), [[-0.5, 0], [-1, 0.5]])
        self.assertEqual(prediction.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(target.tolist(), [[0, 2], [1, 5]])

    def test_plain_prediction_and_trainable_target(self):
        target = Tensor([1, 2], requires_grad=True)
        self.loss_function([3, 0], target).backward()
        self.assertEqual(target.grad.tolist(), [-2, 2])

    def test_shape_mismatch_does_not_broadcast(self):
        for prediction, target in (([1, 2], [[1], [2]]), ([1, 2], 1), ([1, 2], [1, 2, 3])):
            with self.subTest(prediction=prediction, target=target):
                with self.assertRaisesRegex(ValueError, "预测与目标的形状必须相同"):
                    self.loss_function(prediction, target)

    def test_empty_data_has_no_mean(self):
        with self.assertRaisesRegex(ValueError, "归约范围内没有元素"):
            self.loss_function([], [])


if __name__ == "__main__":
    unittest.main()
