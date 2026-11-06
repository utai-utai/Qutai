import itertools
import math
import unittest

from qutai import Tensor, nn
from qutai.backends.device import get_backend
from qutai.core.grad import record
from qutai.lm.config import Config
from qutai.lm.decoder import Decoder
from qutai.lm.training import cosine_learning_rate, train, train_step


def tiny_config(**overrides):
    return Config(
        **{
            **dict(
                vocab_size=9,
                n_layers=1,
                d_model=4,
                n_heads=2,
                n_kv_heads=1,
                d_ff=8,
                max_seq_len=4,
                activation_checkpointing=False,
            ),
            **overrides,
        }
    )


class InfiniteGradient:
    def __init__(self):
        self.weight = Tensor([1.0], requires_grad=True)

    def parameters(self):
        return (self.weight,)

    def loss(self, inputs, labels, **unused):
        return record(
            Tensor(1.0),
            (self.weight,),
            lambda gradient: (get_backend("cpu").full(1, float("inf")),),
        )


class LMTrainingTests(unittest.TestCase):
    def test_warmup_and_cosine_bounds(self):
        schedule = lambda step: cosine_learning_rate(
            step, peak=0.01, total_steps=10, warmup_steps=2, min_ratio=0.1
        )
        self.assertEqual(schedule(0), 0.005)
        self.assertEqual(schedule(1), 0.01)
        self.assertEqual(schedule(2), 0.01)
        self.assertAlmostEqual(schedule(10), 0.001)
        self.assertTrue(all(0.001 <= schedule(step) <= 0.01 for step in range(11)))

    def test_dense_accumulation_matches_combined_batch(self):
        first = Decoder(tiny_config(), dtype="float64", seed=17)
        combined = Decoder(tiny_config(), dtype="float64", seed=17)
        tokens = [[1, 2, 3, 4], [3, 2, 1, 5]]
        labels = [[-100, -100, -100, 4], [-100, 2, 1, 5]]
        optimizer = nn.SGD(first, learning_rate=0.03)
        metrics = train_step(
            first,
            optimizer,
            iter([([tokens[0]], [labels[0]]), ([tokens[1]], [labels[1]])]),
            gradient_accumulation=2,
            max_grad_norm=None,
        )
        reference_optimizer = nn.SGD(combined, learning_rate=0.03)
        loss = combined.loss(tokens, labels)
        loss.backward(retain_graph=False)
        reference_optimizer.step()
        self.assertAlmostEqual(metrics["loss"], loss.item(), places=12)
        self.assertEqual(metrics["tokens"], 6)
        self.assertEqual(metrics["supervised_tokens"], 4)
        for actual, expected in zip(first.parameters(), combined.parameters()):
            for left, right in zip(actual.storage.buffer, expected.storage.buffer):
                self.assertAlmostEqual(left, right, places=12)

    def test_moe_unequal_mask_weights_lm_and_auxiliary_separately(self):
        configuration = tiny_config(
            ffn_type="moe", n_experts=2, top_k=1, d_expert=5, aux_loss_coef=0.17
        )
        actual = Decoder(configuration, dtype="float64", seed=22)
        expected = Decoder(configuration, dtype="float64", seed=22)
        batches = [
            ([[1, 2, 3, 4]], [[-100, -100, -100, 4]]),
            ([[4, 3, 2, 1]], [[-100, 3, 2, 1]]),
        ]
        optimizer = nn.SGD(actual, learning_rate=0.02)
        metrics = train_step(
            actual,
            optimizer,
            iter(batches),
            gradient_accumulation=2,
            max_grad_norm=None,
        )
        reference_optimizer = nn.SGD(expected, learning_rate=0.02)
        reference_loss = 0
        for (inputs, labels), weight in zip(batches, (0.25, 0.75)):
            expected.loss(inputs, labels)
            weighted = (
                expected.last_lm_loss * weight
                + configuration.aux_loss_coef * expected.last_auxiliary_loss / 2
            )
            reference_loss += weighted.item()
            weighted.backward(retain_graph=False)
        reference_optimizer.step()
        self.assertAlmostEqual(metrics["loss"], reference_loss, places=12)
        self.assertAlmostEqual(
            metrics["loss"],
            metrics["lm_loss"]
            + configuration.aux_loss_coef * metrics["auxiliary_loss"],
            places=12,
        )
        for left, right in zip(actual.parameters(), expected.parameters()):
            for actual_value, expected_value in zip(
                left.storage.buffer, right.storage.buffer
            ):
                self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_train_loop_global_steps_and_tokens(self):
        model = Decoder(tiny_config(), seed=2)
        optimizer = nn.AdamW(model, learning_rate=0.005)
        batches = itertools.cycle([([[1, 2, 3, 4]], None)])
        metrics = list(
            train(
                model,
                optimizer,
                batches,
                steps=3,
                start_step=5,
                tokens_seen=20,
                schedule=lambda step: 0.01 / (step + 1),
            )
        )
        self.assertEqual([item["global_step"] for item in metrics], [6, 7, 8])
        self.assertEqual(metrics[-1]["tokens_seen"], 29)
        self.assertEqual(metrics[-1]["learning_rate"], 0.01 / 8)
        self.assertTrue(all(math.isfinite(item["loss"]) for item in metrics))

    def test_finite_gradient_check_still_runs_without_clipping(self):
        model = InfiniteGradient()
        optimizer = nn.SGD(model)
        with self.assertRaises(ValueError):
            train_step(model, optimizer, iter([([[0, 1]], None)]), max_grad_norm=None)
        self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
