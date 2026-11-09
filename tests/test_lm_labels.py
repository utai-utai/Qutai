from array import array
import math
import unittest
from unittest.mock import patch

from qutai import is_available, lm
from qutai.backends import cpu, cpu_lm, cuda, cuda_lm


class HeadLabelTests(unittest.TestCase):
    def test_cpu_offsets_keep_masked_loss_and_gradient(self):
        rows, vocab = 3, 4
        values = [0.1 * i for i in range(rows * vocab)]
        labels = (3, 3, 1, -100, 2, 0)
        selected = labels[2:5]
        logits = cpu.storage(values, dtype="float64")
        loss = cpu_lm.head_loss(logits, labels, rows, vocab, start=2).tolist()[0]
        expected = sum(
            math.log(sum(math.exp(v) for v in values[row * vocab : (row + 1) * vocab]))
            - values[row * vocab + target]
            for row, target in enumerate(selected)
            if target != -100
        )
        self.assertAlmostEqual(loss, expected)
        actual = cpu_lm.head_gradient(
            logits.copy(), labels, rows, vocab, cpu.storage([1.0]), 2, start=2
        ).tolist()
        reference = cpu_lm.head_gradient(
            logits.copy(), selected, rows, vocab, cpu.storage([1.0]), 2
        ).tolist()
        self.assertEqual(actual, reference)


@unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
class CUDAHeadLabelTests(unittest.TestCase):
    def test_targets_upload_once_and_reused_in_backward(self):
        config = lm.Config(
            vocab_size=128,
            n_layers=1,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
            max_seq_len=33,
        )
        reference = lm.Decoder(config, dtype="float32", seed=7)
        model = lm.Decoder(config, device="cuda", seed=8)
        for expected, actual in zip(
            reference.parameters(), model.parameters(), strict=True
        ):
            cuda.copy_into(
                actual.storage, cuda.storage(expected.storage.tolist(), dtype="float32")
            )
        tokens = [list(range(1, 34)), list(range(34, 67))]
        labels = [[-100] * 4 + row[4:] for row in tokens]
        expected = reference.loss(tokens, labels, chunk_size=17)
        expected.backward(retain_graph=False)
        uploads = []
        runtime = cuda.get_runtime()
        original = runtime.upload

        def upload(pointer, buffer):
            uploads.append(
                (buffer.typecode if isinstance(buffer, array) else None, len(buffer))
            )
            return original(pointer, buffer)

        with patch.object(runtime, "upload", side_effect=upload):
            actual = model.loss(tokens, labels, chunk_size=17)
            actual.backward(retain_graph=False)
        self.assertEqual(uploads.count(("q", 64)), 1)
        # 其余最多四次为 embedding 的输入与确定性反向索引，不随 head 分块数增长。
        self.assertLessEqual(len(uploads), 5)
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)
        for expected, actual in zip(
            reference.parameters(), model.parameters(), strict=True
        ):
            self.assertLess(
                max(
                    abs(a - b)
                    for a, b in zip(
                        expected.grad.storage.tolist(), actual.grad.storage.tolist()
                    )
                ),
                2e-5,
            )

    def test_scalar_inputs_enqueue_fill_without_host_upload(self):
        with patch.object(
            cuda.get_runtime(), "upload", side_effect=AssertionError("标量无需HtoD复制")
        ):
            values = [
                cuda.storage([1.25], dtype=dtype) for dtype in ("float32", "float64")
            ]
            integer = cuda.storage([-7], dtype="int64")
        self.assertEqual([value.tolist() for value in values], [[1.25], [1.25]])
        self.assertEqual(integer.tolist(), [-7])

    def test_expert_addresses_do_not_use_host_upload(self):
        values = [0, 1, -1, 2**63 - 1, -(2**63), 3, 7, 9, 11, 13, 17]
        with patch.object(
            cuda.get_runtime(), "upload", side_effect=AssertionError("地址表无需HtoD")
        ):
            table = cuda_lm._pointer_table(values)
        self.assertEqual(table.tolist(), values)
