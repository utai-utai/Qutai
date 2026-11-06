"""稀疏专家的路由、辅助损失与前向/反向对照。"""

import unittest
from unittest.mock import patch

from qutai import nn
from qutai.backends import cpu_lm, cuda, cuda_lm
from qutai.lm.config import Config
from qutai.lm.decoder import Decoder


def config(**changes):
    values = dict(
        vocab_size=17,
        n_layers=2,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=12,
        max_seq_len=8,
        ffn_type="moe",
        n_experts=3,
        top_k=2,
        d_expert=6,
    )
    values.update(changes)
    return Config(**values)


class MoETests(unittest.TestCase):
    def test_router_auxiliary_and_expert_parameter_gradients(self):
        model = Decoder(
            config(n_layers=1, activation_checkpointing=False), dtype="float64", seed=11
        )
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        loss = model.loss(tokens, chunk_size=3)
        loss.backward()
        parameter = model.blocks[0].ffn.router.weight
        for index in (0, 5, 12):
            original = parameter.storage[index]
            parameter.storage[index] = original + 1e-5
            upper = model.loss(tokens, train=False).item()
            parameter.storage[index] = original - 1e-5
            lower = model.loss(tokens, train=False).item()
            parameter.storage[index] = original
            self.assertAlmostEqual(
                parameter.grad.storage[index], (upper - lower) / 2e-5, places=7
            )
        self.assertAlmostEqual(
            sum(model.blocks[0].ffn.utilization.storage), 1, places=12
        )

    def test_checkpoint_preserves_auxiliary_gradient(self):
        plain = Decoder(config(activation_checkpointing=False), dtype="float64", seed=2)
        checked = Decoder(
            config(activation_checkpointing=True), dtype="float64", seed=2
        )
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        plain.loss(tokens).backward()
        checked.loss(tokens).backward()
        for a, b in zip(plain.parameters(), checked.parameters()):
            if a.grad is None and b.grad is None:
                continue
            for x, y in zip(a.grad.storage, b.grad.storage):
                self.assertAlmostEqual(x, y, places=11)

    def test_routes_are_sparse_and_weights_normalized(self):
        scores = cpu_lm.cpu.storage([3.0, 2.0, -1.0, -1.0, 2.0, 3.0], dtype="float64")
        route = cpu_lm.routing(scores, 2, 3, 2)
        self.assertEqual(route["indices"].tolist(), [0, 1, 2, 1])
        self.assertEqual(route["counts"], (1, 2, 1))
        self.assertAlmostEqual(sum(route["weights"][:2]), 1)
        self.assertAlmostEqual(sum(route["weights"][2:]), 1)


    def test_empty_expert_has_none_gradient_and_keeps_adam_state(self):
        model = Decoder(config(n_layers=1, top_k=1), dtype='float64', seed=8)
        block = model.blocks[0]
        for i in range(block.attn.out.weight.size):
            block.attn.out.weight.storage[i] = 0
        for j in range(model.config.d_model):
            model.embedding.weight.storage[model.config.d_model + j] = .1

        def select(expert):
            for row in range(model.config.n_experts):
                for j in range(model.config.d_model):
                    block.ffn.router.weight.storage[row * model.config.d_model + j] = .3 if row == expert else -.1

        optimizer = nn.AdamW(model, learning_rate=.001, weight_decay=.1)
        select(1)
        model.loss([[1, 1, 1, 1]]).backward(retain_graph=False)
        self.assertTrue(all(p.grad is None for p in block.ffn.experts[0].parameters()))
        optimizer.step()
        preserved = [(p.storage.tolist(), optimizer.averages[p].tolist(), optimizer.square_averages[p].tolist(), optimizer.steps[p]) for p in block.ffn.experts[1].parameters()]
        optimizer.zero_grad()
        select(0)
        model.loss([[1, 1, 1, 1]]).backward(retain_graph=False)
        self.assertTrue(all(p.grad is None for p in block.ffn.experts[1].parameters()))
        self.assertTrue(any(abs(value) > 0 for value in block.ffn.router.weight.grad.storage))
        optimizer.step()
        for parameter, expected in zip(block.ffn.experts[1].parameters(), preserved):
            self.assertEqual((parameter.storage.tolist(), optimizer.averages[parameter].tolist(), optimizer.square_averages[parameter].tolist(), optimizer.steps[parameter]), expected)


@unittest.skipUnless(cuda.is_available(), "需要真实 CUDA 设备")
class CUDAMoETests(unittest.TestCase):
    def test_gpu_sparse_routes_and_two_updates_match_cpu(self):
        source = Decoder(config(), seed=19)
        target = Decoder(config(), device="cuda", seed=20)
        for a, b in zip(source.parameters(), target.parameters()):
            cuda.copy_into(b.storage, cuda.storage(a.storage.tolist(), dtype="float32"))
        tokens = [[1, 2, 3, 4], [2, 3, 4, 5]]
        optimizers = [nn.AdamW(m, learning_rate=0.001) for m in (source, target)]
        for _ in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad()
            reference = source.loss(tokens, chunk_size=3)
            reference.backward()
            with (
                patch.object(
                    cuda.CUDAStorage,
                    "tolist",
                    side_effect=AssertionError("不能导出GPU数值或路由索引"),
                ),
                patch.object(
                    cpu_lm, "routing", side_effect=AssertionError("路由不能回退CPU")
                ),
            ):
                actual = target.loss(tokens, chunk_size=3)
                actual.backward()
                optimizers[1].step()
            optimizers[0].step()
            self.assertAlmostEqual(reference.item(), actual.item(), places=5)
            for a, b in zip(source.parameters(), target.parameters()):
                if a.grad is None and b.grad is None:
                    continue
                for x, y in zip(a.grad.storage.tolist(), b.grad.storage.tolist()):
                    self.assertLessEqual(abs(x - y), 2e-5 * (1 + abs(y)))
                for x, y in zip(a.storage.tolist(), b.storage.tolist()):
                    self.assertLessEqual(abs(x - y), 2e-5 * (1 + abs(y)))

    def test_gpu_route_pack_has_stable_token_slot_order_across_blocks(self):
        rows, experts, top_k = 517, 5, 3
        scores = [
            ((row * 37 + expert * 11) % 101) / 13
            for row in range(rows)
            for expert in range(experts)
        ]
        reference = cpu_lm.routing(
            cpu_lm.cpu.storage(scores, dtype="float32"), rows, experts, top_k
        )
        values = cuda.storage(scores, dtype="float32")
        for _ in range(3):
            route = cuda_lm.routing(values, rows, experts, top_k)
            self.assertEqual(route["counts"], reference["counts"])
            for expert, count in enumerate(route["counts"]):
                ids = cuda_lm.take(
                    route["ids"], expert * route["capacity"], count
                ).tolist()
                slots = cuda_lm.take(
                    route["slots"], expert * route["capacity"], count
                ).tolist()
                self.assertEqual(ids, reference["ids"][expert])
                self.assertEqual(slots, reference["slots"][expert])
