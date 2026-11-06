import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qutai import is_available, nn
from qutai.backends.cpu import CPUStorage
from qutai.backends.cuda import CUDAStorage
from qutai.lm import checkpoint
from qutai.lm.checkpoint import load_checkpoint, save_checkpoint
from qutai.lm.data import PackedTokens, StatefulBatcher, TokenShards, write_token_shards


class SharedModel:
    def __init__(self, *, dtype="float32"):
        self.layer = nn.Linear(3, 3, seed=41).to(dtype=dtype)
        self.network = nn.Model([self.layer, nn.ReLU(), self.layer])

    def parameters(self):
        return self.network.parameters()

    def named_parameters(self):
        return (
            ("first.weight", self.layer.weight),
            ("first.bias", self.layer.bias),
            ("second.weight", self.layer.weight),
            ("second.bias", self.layer.bias),
        )

    def __call__(self, values):
        return self.network(values)


def update(model, optimizer, batches, rng):
    inputs, _ = batches.next()
    targets = [
        [
            value * 0.1 + rng.uniform(-0.1, 0.1) + random.uniform(-0.01, 0.01)
            for value in row
        ]
        for row in inputs
    ]
    optimizer.zero_grad()
    loss = nn.MSE()(model(inputs), targets)
    loss.backward(retain_graph=False)
    optimizer.step()
    return loss.item()


class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_with_groups_rng_cursor_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = write_token_shards(
                Path(directory) / "data",
                (index % 7 for index in range(61)),
                shard_tokens=8,
            )
            with TokenShards(corpus) as tokens:
                dataset = PackedTokens(tokens, 3)
                batches = StatefulBatcher(dataset, batch_size=2, seed=47, repeat=True)
                model = SharedModel()
                weight, bias = model.parameters()
                optimizer = nn.AdamW(
                    [
                        {"params": [weight], "weight_decay": 0.03},
                        {"params": [bias], "learning_rate": 0.002, "weight_decay": 0},
                    ],
                    learning_rate=0.005,
                )
                rng = random.Random(71)
                random.seed(91)
                for _ in range(2):
                    update(model, optimizer, batches, rng)
                path = Path(directory) / "checkpoint"
                with patch.object(
                    CPUStorage,
                    "tolist",
                    side_effect=AssertionError("禁止把全部权重转列表"),
                ):
                    save_checkpoint(
                        path,
                        model,
                        optimizer,
                        global_step=2,
                        tokens=8,
                        learning_rate=0.005,
                        data=batches,
                        rngs={"samples": rng},
                        state={"accumulation_step": 0},
                        shard_bytes=31,
                        chunk_bytes=12,
                    )
                expected_losses = [
                    update(model, optimizer, batches, rng) for _ in range(4)
                ]
                expected_values = [
                    parameter.storage.buffer.tobytes()
                    for parameter in model.parameters()
                ]
                expected_averages = [
                    optimizer.averages[parameter].buffer.tobytes()
                    for parameter in model.parameters()
                ]
                expected_rng, expected_python = rng.getstate(), random.getstate()
                expected_data = batches.state_dict()
                restored = SharedModel()
                resumed_optimizer = nn.AdamW(restored, learning_rate=0.8)
                restored_batches = StatefulBatcher(dataset, batch_size=1, seed=2)
                restored_rng = random.Random(3)
                identities = [
                    (id(parameter), id(parameter.storage))
                    for parameter in restored.parameters()
                ]
                with patch.object(
                    CPUStorage,
                    "tolist",
                    side_effect=AssertionError("禁止把全部权重转列表"),
                ):
                    state = load_checkpoint(
                        path,
                        restored,
                        resumed_optimizer,
                        data=restored_batches,
                        rngs={"samples": restored_rng},
                        chunk_bytes=7,
                    )
                self.assertEqual(state["global_step"], 2)
                self.assertEqual(state["extra"]["accumulation_step"], 0)
                self.assertEqual(
                    identities,
                    [
                        (id(parameter), id(parameter.storage))
                        for parameter in restored.parameters()
                    ],
                )
                named = dict(restored.named_parameters())
                self.assertIs(named["first.weight"], named["second.weight"])
                self.assertEqual(
                    [
                        update(
                            restored, resumed_optimizer, restored_batches, restored_rng
                        )
                        for _ in range(4)
                    ],
                    expected_losses,
                )
                self.assertEqual(
                    [
                        parameter.storage.buffer.tobytes()
                        for parameter in restored.parameters()
                    ],
                    expected_values,
                )
                self.assertEqual(
                    [
                        resumed_optimizer.averages[parameter].buffer.tobytes()
                        for parameter in restored.parameters()
                    ],
                    expected_averages,
                )
                self.assertEqual(restored_rng.getstate(), expected_rng)
                self.assertEqual(random.getstate(), expected_python)
                self.assertEqual(restored_batches.state_dict(), expected_data)

    def test_failed_new_generation_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SharedModel()
            manifest = save_checkpoint(directory, model, shard_bytes=17, chunk_bytes=8)
            original = {
                path.name: path.read_bytes() for path in Path(directory).iterdir()
            }
            chunks = checkpoint._storage_chunks

            def failing(storage, size):
                yield next(chunks(storage, size))
                raise OSError("模拟写入中断")

            with (
                patch.object(checkpoint, "_storage_chunks", failing),
                self.assertRaises(OSError),
            ):
                save_checkpoint(directory, model, shard_bytes=17, chunk_bytes=8)
            self.assertEqual(
                original,
                {path.name: path.read_bytes() for path in Path(directory).iterdir()},
            )
            load_checkpoint(manifest, model)

    def test_manifest_publish_failure_keeps_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SharedModel()
            save_checkpoint(directory, model, shard_bytes=29)
            original = {
                path.name: path.read_bytes() for path in Path(directory).iterdir()
            }
            replace = checkpoint.os.replace

            def failing(source, target):
                if Path(source).name.startswith("manifest-"):
                    raise OSError("模拟 manifest 原子发布失败")
                return replace(source, target)

            with (
                patch.object(checkpoint.os, "replace", failing),
                self.assertRaises(OSError),
            ):
                save_checkpoint(directory, model, shard_bytes=29)
            self.assertEqual(
                original,
                {path.name: path.read_bytes() for path in Path(directory).iterdir()},
            )

    def test_invalid_optimizer_metadata_fails_before_parameter_write(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SharedModel()
            optimizer = nn.AdamW(model)
            manifest = save_checkpoint(directory, model, optimizer)
            metadata = json.loads(manifest.read_text())
            metadata["optimizer"]["steps"]["parameter.0"] = -1
            manifest.write_text(json.dumps(metadata))
            model.layer.weight.storage.buffer[0] = 91
            before = model.layer.weight.storage.buffer.tobytes()
            with self.assertRaises(ValueError):
                load_checkpoint(manifest, model, optimizer)
            self.assertEqual(model.layer.weight.storage.buffer.tobytes(), before)

    def test_corruption_or_shared_relationship_fails_before_parameter_write(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SharedModel()
            manifest = save_checkpoint(directory, model, shard_bytes=19)
            before = [
                parameter.storage.buffer.tobytes() for parameter in model.parameters()
            ]
            metadata = json.loads(manifest.read_text())
            shard = Path(directory) / metadata["shards"][0]["file"]
            content = bytearray(shard.read_bytes())
            content[0] ^= 1
            shard.write_bytes(content)
            with self.assertRaises(ValueError):
                load_checkpoint(directory, model)
            self.assertEqual(
                before,
                [
                    parameter.storage.buffer.tobytes()
                    for parameter in model.parameters()
                ],
            )

    def test_float64_gradients_and_optimizer_storage_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Model([nn.Linear(2, 1, seed=17)])
            optimizer = nn.AdamW(model)
            loss = nn.MSE()(model([[1, 2]]), [[3]])
            loss.backward(retain_graph=False)
            optimizer.step()
            expected = [
                parameter.storage.buffer.tobytes() for parameter in model.parameters()
            ]
            gradients = [
                parameter.grad.storage.buffer.tobytes()
                for parameter in model.parameters()
            ]
            ids = [
                id(optimizer.averages[parameter]) for parameter in model.parameters()
            ]
            save_checkpoint(
                directory,
                model,
                optimizer,
                state={"accumulation_step": 1},
                shard_bytes=5,
                chunk_bytes=3,
            )
            optimizer.zero_grad()
            for parameter in model.parameters():
                parameter.storage.buffer[0] = 99
            load_checkpoint(directory, model, optimizer, chunk_bytes=3)
            self.assertEqual(
                [
                    parameter.storage.buffer.tobytes()
                    for parameter in model.parameters()
                ],
                expected,
            )
            self.assertEqual(
                [
                    parameter.grad.storage.buffer.tobytes()
                    for parameter in model.parameters()
                ],
                gradients,
            )
            self.assertEqual(
                [id(optimizer.averages[parameter]) for parameter in model.parameters()],
                ids,
            )

    def test_batch_norm_buffers_and_tokenizer_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Model([nn.BatchNorm2d(1)])
            model([[[[1, 3], [5, 7]]]])
            expected = model.layers[0].running_mean.buffer.tobytes()
            asset = Path(directory) / "tokenizer.asset"
            asset.write_bytes(b"fixed vocabulary")
            path = Path(directory) / "checkpoint"
            save_checkpoint(path, model, tokenizer=asset)
            model.layers[0].running_mean.buffer[0] = 88
            load_checkpoint(path, model, tokenizer=asset)
            self.assertEqual(model.layers[0].running_mean.buffer.tobytes(), expected)
            asset.write_bytes(b"different vocabulary")
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, tokenizer=asset)

    @unittest.skipUnless(is_available("cuda"), "需要 CUDA 实卡")
    def test_cuda_streamed_restore_matches_next_adam_step_without_tolist(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Model(
                [nn.Linear(3, 5, seed=7), nn.ReLU(), nn.Linear(5, 2, seed=8)]
            ).to("cuda", dtype="float32")
            optimizer = nn.AdamW(model, learning_rate=0.003)

            def step(network, update):
                update.zero_grad()
                loss = nn.MSE()(network([[1, 2, 3], [3, 1, -2]]), [[0.5, 1], [2, 0]])
                loss.backward(retain_graph=False)
                update.step()
                return loss.item()

            step(model, optimizer)
            step(model, optimizer)
            with patch.object(
                CUDAStorage,
                "tolist",
                side_effect=AssertionError("checkpoint 禁止显存全量 tolist"),
            ):
                save_checkpoint(
                    directory,
                    model,
                    optimizer,
                    global_step=2,
                    shard_bytes=43,
                    chunk_bytes=11,
                )
            expected_loss = step(model, optimizer)
            expected = [parameter.tolist() for parameter in model.parameters()]
            averages = [
                optimizer.averages[parameter].tolist()
                for parameter in model.parameters()
            ]
            target = nn.Model(
                [nn.Linear(3, 5, seed=91), nn.ReLU(), nn.Linear(5, 2, seed=92)]
            ).to("cuda", dtype="float32")
            restored_optimizer = nn.AdamW(target, learning_rate=0.8)
            step(target, restored_optimizer)
            identities = [
                (
                    id(parameter),
                    id(parameter.storage),
                    id(restored_optimizer.averages[parameter]),
                )
                for parameter in target.parameters()
            ]
            with patch.object(
                CUDAStorage,
                "tolist",
                side_effect=AssertionError("checkpoint 禁止显存全量 tolist"),
            ):
                load_checkpoint(directory, target, restored_optimizer, chunk_bytes=7)
            self.assertEqual(
                identities,
                [
                    (
                        id(parameter),
                        id(parameter.storage),
                        id(restored_optimizer.averages[parameter]),
                    )
                    for parameter in target.parameters()
                ],
            )
            self.assertEqual(step(target, restored_optimizer), expected_loss)
            self.assertEqual(
                [parameter.tolist() for parameter in target.parameters()], expected
            )
            self.assertEqual(
                [
                    restored_optimizer.averages[parameter].tolist()
                    for parameter in target.parameters()
                ],
                averages,
            )


if __name__ == "__main__":
    unittest.main()
