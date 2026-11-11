import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Tensor, is_available, nn


class DirectLayerTests(unittest.TestCase):
    def test_direct_tensor_inputs_keep_device_and_backward(self):
        for device in ('cpu', 'apple'):
            if not is_available(device): continue
            for kind, options, shape in (
                (nn.Flatten, {}, (1, 4)),
                (nn.GlobalAvgPool2d, {}, (1, 1)),
                (nn.MaxPool2d, {'kernel_size': 2}, (1, 1, 1, 1)),
                (nn.ReLU, {}, (1, 1, 2, 2)),
                (nn.GELU, {}, (1, 1, 2, 2)),
                (nn.Softmax, {'axis': -1}, (1, 1, 2, 2)),
                (nn.Mean, {'axis': -1}, (1, 1, 2)),
                (nn.Identity, {}, (1, 1, 2, 2)),
            ):
                with self.subTest(device=device, layer=kind.__name__):
                    x = Tensor([[[[1, -2], [3, 4]]]], device=device, requires_grad=True)
                    result = kind(x, **options)
                    expected = kind(**options)(x)
                    self.assertEqual((result.shape, result.device), (shape, device))
                    self.assertEqual(result.tolist(), expected.tolist())
                    result.sum().backward()
                    self.assertEqual(x.grad.device, device)

    def test_plain_data_and_configuration_are_unambiguous(self):
        self.assertEqual(nn.Flatten([[1, 2], [3, 4]], start_dim=0).tolist(), [1, 2, 3, 4])
        self.assertEqual(nn.Flatten(data=3, start_dim=0).tolist(), [3])
        self.assertEqual(nn.ReLU(-2).item(), 0)
        self.assertEqual(nn.Identity((1, 2)).tolist(), [1, 2])
        self.assertEqual(nn.Softmax([[0, 0]]).tolist(), [[0.5, 0.5]])
        self.assertIsInstance(nn.Flatten(-2), nn.Flatten)
        self.assertEqual(nn.Flatten(0)([[1, 2]]).tolist(), [1, 2])
        self.assertEqual(nn.MaxPool2d((2, 2))([[[[1, 2], [3, 4]]]]).tolist(), [[[[4]]]])
        self.assertEqual(nn.Mean(1)([[1, 3]]).tolist(), [2])
        self.assertEqual(nn.ReLU([1, -2], device='cpu', dtype='float32').dtype, 'float32')

    def test_model_composition_and_serialization_keep_real_layer_objects(self):
        model = nn.Model([nn.ReLU() + nn.Identity(), nn.Flatten(), nn.Linear(4, 1)])
        inputs = [[[1, -2], [3, 4]]]
        result = model(inputs)
        result.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            model.save(path)
            restored = nn.Model().load(path)
            self.assertEqual(restored(inputs, train=False).tolist(), model(inputs, train=False).tolist())
            self.assertIsInstance(restored.layers[1], nn.Flatten)
