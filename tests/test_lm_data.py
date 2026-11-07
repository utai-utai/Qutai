import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qutai import Stream
from qutai.lm.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def test_unicode_whitespace_order_and_integer_roundtrip(self):
        text = "量子 AI\n量😊\t子!"
        tokenizer = Tokenizer(text)
        self.assertEqual(tokenizer.characters, "量子 AI\n😊\t!")
        self.assertEqual(tokenizer.size, 9)
        self.assertEqual(tokenizer.indices, {'量': 0, '子': 1, ' ': 2, 'A': 3, 'I': 4, '\n': 5, '😊': 6, '\t': 7, '!': 8})
        ids = tokenizer.encode(text)
        self.assertEqual(ids, [0, 1, 2, 3, 4, 5, 0, 6, 7, 1, 8])
        self.assertTrue(all(type(index) is int for index in ids))
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.decode(tuple(ids)), text)

    def test_saved_vocabulary_rebuilds_without_renumbering(self):
        source = Tokenizer("z 量a\nz😊a量")
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'vocabulary.json'
            path.write_text(json.dumps({'characters': source.characters}, ensure_ascii=False), encoding='utf-8')
            restored = Tokenizer(json.loads(path.read_text(encoding='utf-8'))['characters'])
        self.assertEqual(restored.characters, source.characters)
        self.assertEqual(restored.indices, source.indices)
        self.assertEqual(restored.encode("😊\nza量 "), source.encode("😊\nza量 "))

    def test_empty_and_invalid_characters_or_ids(self):
        empty = Tokenizer("")
        self.assertEqual(empty.size, 0)
        self.assertEqual(empty.encode(""), [])
        self.assertEqual(empty.decode([]), "")
        tokenizer = Tokenizer("ab")
        for source, text in ((empty, 'a'), (tokenizer, 'c')):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    source.encode(text)
        for ids in ([-1], [2], [0.0], [True], ['1']):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                    tokenizer.decode(ids)


class LanguageBatchTests(unittest.TestCase):
    def test_shifted_nonoverlapping_windows_and_partial_last_batch(self):
        tokens = list(range(12))
        self.assertEqual(list(Stream(tokens, context_length=3, batch_size=2, shuffle=False)), [
            ([[0, 1, 2], [3, 4, 5]], [[1, 2, 3], [4, 5, 6]]),
            ([[6, 7, 8]], [[7, 8, 9]]),
        ])
        self.assertEqual(tokens, list(range(12)))

    def test_boundary_lengths_require_one_extra_target_token(self):
        for length, count in ((0, 0), (2, 0), (3, 0), (4, 1), (6, 1), (7, 2)):
            with self.subTest(length=length):
                result = list(Stream(tuple(range(length)), context_length=3, batch_size=4))
                self.assertEqual(sum(len(inputs) for inputs, _ in result), count)
                for inputs, targets in result:
                    for source, target in zip(inputs, targets, strict=True):
                        self.assertEqual(len(source), 3)
                        self.assertEqual(len(target), 3)
                        self.assertEqual(source[1:], target[:-1])
                        self.assertLess(target[-1], length)

    def test_tuple_tokens_remain_integer_lists(self):
        result = list(Stream(tuple(range(1000, 1007)), context_length=3, shuffle=False))
        self.assertEqual(result, [([[1000, 1001, 1002], [1003, 1004, 1005]], [[1001, 1002, 1003], [1004, 1005, 1006]])])
        self.assertTrue(all(type(index) is int for inputs, targets in result for rows in (inputs, targets) for row in rows for index in row))
        self.assertTrue(all(type(row) is list for inputs, targets in result for rows in (inputs, targets) for row in rows))

    def test_shuffle_is_reproducible_keeps_pairs_and_preserves_random_state(self):
        tokens = list(range(31))
        state = random.getstate()
        first = list(Stream(tokens, context_length=3, batch_size=4, shuffle=True, seed=42))
        second = list(Stream(tokens, context_length=3, batch_size=4, shuffle=True, seed=42))
        other = list(Stream(tokens, context_length=3, batch_size=4, shuffle=True, seed=7))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(random.getstate(), state)
        self.assertEqual([len(inputs) for inputs, _ in first], [4, 4, 2])
        ordered = list(Stream(tokens, context_length=3, batch_size=4, shuffle=False))
        self.assertEqual(sorted((source, target) for inputs, targets in first for source, target in zip(inputs, targets, strict=True)),
                         [(source, target) for inputs, targets in ordered for source, target in zip(inputs, targets, strict=True)])
        self.assertEqual(tokens, list(range(31)))

    def test_context_and_batch_size_must_be_positive_integers(self):
        for name in ('context_length', 'batch_size'):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, '[\u4e00-\u9fff]'):
                        list(Stream([0, 1, 2], **{name: value}))


if __name__ == '__main__':
    unittest.main()
