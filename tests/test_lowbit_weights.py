import io
import unittest
import torch
from src.lowbit_weights import pack_weight, unpack_weight, pack_model, restore_model


class PackedWeightTests(unittest.TestCase):
    def test_packed_padding_and_quantization_error(self):
        w = torch.tensor([[0., 1., -1., .33, -.2], [0., 0., 0., 0., 0.]])
        for bits in (2, 4):
            record = pack_weight(w, bits)
            self.assertEqual(record['packed'].numel(), (w.numel() * bits + 7) // 8)
            restored = unpack_weight(record)
            self.assertEqual(restored.shape, w.shape)
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(torch.all((restored - w).abs() <= record['scale'][:, None] / 2 + 1e-6))
            self.assertTrue(torch.equal(restored[1], w[1]))

    def test_conv_linear_roundtrip_and_remaining_state(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4),
                                    torch.nn.Flatten(), torch.nn.Linear(16, 2)).eval()
        x = torch.randn(2, 3, 4, 4)
        for bits in (2, 4):
            payload = pack_model(model, bits)
            self.assertEqual(set(payload['packed_weights']), {'0.weight', '3.weight'})
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            restored = restore_model(__import__('copy').deepcopy(model), saved)
            self.assertTrue(torch.equal(restored[1].running_var, model[1].running_var))
            self.assertTrue(torch.equal(restored[0].bias, model[0].bias))
            self.assertEqual(restored(x).shape, (2, 2))
            self.assertTrue(torch.isfinite(restored(x)).all())


if __name__ == '__main__':
    unittest.main()
