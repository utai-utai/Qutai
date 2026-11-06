import json
import tempfile
import unittest
from pathlib import Path

from qutai.lm.data import PackedTokens, StatefulBatcher, TokenShards, write_token_shards


class TokenShardTests(unittest.TestCase):
    def test_uint16_uint32_cross_shard_slices_and_read_only(self):
        for dtype, values in (
            ("uint16", list(range(41))),
            ("uint32", [65536 + i for i in range(41)]),
        ):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                manifest = write_token_shards(
                    directory, iter(values), dtype=dtype, shard_tokens=7
                )
                original = {
                    path.name: path.read_bytes()
                    for path in Path(directory).glob("*.bin")
                }
                with TokenShards(manifest, verify=True, max_open=2) as tokens:
                    self.assertEqual(tokens[:], values)
                    self.assertEqual(tokens[3:26], values[3:26])
                    self.assertEqual(tokens[::-3], values[::-3])
                    self.assertEqual(tokens[-1], values[-1])
                    self.assertLessEqual(len(tokens._maps), 2)
                self.assertEqual(
                    original,
                    {
                        path.name: path.read_bytes()
                        for path in Path(directory).glob("*.bin")
                    },
                )

    def test_utailm_manifest_and_internal_shift_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_token_shards(directory, range(13), shard_tokens=3)
            data = json.loads(manifest.read_text())
            data["shards"] = {"train": data["shards"]}
            manifest.write_text(json.dumps(data))
            with TokenShards(directory, split="train") as tokens:
                dataset = PackedTokens(tokens, sequence_length=5)
                self.assertEqual(len(dataset), 3)
                first, labels = dataset[0]
                self.assertEqual(first, [0, 1, 2, 3, 4])
                self.assertEqual(labels, first)
                self.assertIsNot(first, labels)
                self.assertEqual(dataset[1][0], [4, 5, 6, 7, 8])

    def test_cursor_rng_restore_across_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_token_shards(directory, range(101), shard_tokens=9)
            with TokenShards(manifest) as tokens:
                dataset = PackedTokens(tokens, 5)
                first = StatefulBatcher(dataset, batch_size=3, seed=19, repeat=True)
                for _ in range(7):
                    first.next()
                state = json.loads(json.dumps(first.state_dict()))
                expected = [first.next() for _ in range(20)]
                restored = StatefulBatcher(dataset, seed=99)
                restored.load_state_dict(state)
                self.assertEqual([restored.next() for _ in range(20)], expected)
                self.assertEqual(restored.state_dict(), first.state_dict())

    def test_epoch_is_a_permutation_and_repeat_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_token_shards(directory, range(41))
            with TokenShards(manifest) as tokens:
                data = StatefulBatcher(PackedTokens(tokens, 5), seed=71)
                starts = [data.next()[0][0][0] for _ in range(10)]
                self.assertEqual(sorted(starts), list(range(0, 40, 4)))
                with self.assertRaises(StopIteration):
                    data.next()

    def test_changed_corpus_or_window_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_token_shards(directory, range(41))
            with TokenShards(manifest) as tokens:
                state = StatefulBatcher(PackedTokens(tokens, 5)).state_dict()
                target = StatefulBatcher(PackedTokens(tokens, 6))
                with self.assertRaises(ValueError):
                    target.load_state_dict(state)
            shard = next(Path(directory).glob("*.bin"))
            content = bytearray(shard.read_bytes())
            content[0] ^= 1
            shard.write_bytes(content)
            with self.assertRaises(ValueError):
                TokenShards(manifest, verify=True)

    def test_invalid_stream_does_not_publish_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_token_shards(directory, [1, 2, 3, 65536], shard_tokens=2)
            self.assertEqual(list(Path(directory).iterdir()), [])
            manifest = write_token_shards(directory, [1, 2, 3])
            before = manifest.read_bytes()
            with self.assertRaises(FileExistsError):
                write_token_shards(directory, [9])
            self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
