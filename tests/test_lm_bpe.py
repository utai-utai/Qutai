import json
import tempfile
import unittest
from pathlib import Path

from qutai.lm.bpe import ByteBPE, SPECIAL_TOKENS


def tokenizer_fixture(path):
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {value: chr(value) for value in values}
    extra = 256
    for value in range(256):
        if value not in alphabet:
            alphabet[value] = chr(extra)
            extra += 1
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab.update({alphabet[value]: len(SPECIAL_TOKENS) + value for value in range(256)})
    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    configuration = {
        "normalizer": None,
        "post_processor": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "use_regex": True,
        },
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": index, "content": token, "special": True}
            for index, token in enumerate(SPECIAL_TOKENS)
        ],
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }
    path.write_text(json.dumps(configuration), encoding="utf8")
    return configuration


class ByteBPETests(unittest.TestCase):
    def test_byte_roundtrip_and_ranked_merges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            configuration = tokenizer_fixture(path)
            tokenizer = ByteBPE(path)
            self.assertEqual(
                tokenizer.encode("hello"), [configuration["model"]["vocab"]["hello"]]
            )
            for text in (
                "你好🙂 café é",
                "I'm you've\n  hello\t世界",
                "\x00\x1c\u0085",
                "",
            ):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_ids_and_explicit_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer_fixture(path)
            tokenizer = ByteBPE.from_file(path)
            self.assertEqual(tokenizer.encode("".join(SPECIAL_TOKENS)), list(range(10)))
            self.assertEqual(tokenizer.decode(list(range(10))), "")
            self.assertEqual(
                tokenizer.decode(list(range(10)), skip_special_tokens=False),
                "".join(SPECIAL_TOKENS),
            )
            self.assertEqual(
                tokenizer.encode("hello", add_special_tokens=True),
                tokenizer.encode("hello", add_special_tokens=False),
            )
            self.assertEqual(tokenizer.token_to_id("<|eos|>"), 2)
            self.assertEqual(len(tokenizer.sha256), 64)
            self.assertEqual(tokenizer.hash, tokenizer.sha256)

    def test_unknown_configuration_and_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            configuration = tokenizer_fixture(path)
            tokenizer = ByteBPE(path)
            with self.assertRaises(ValueError):
                tokenizer.decode([-1])
            configuration["normalizer"] = {"type": "Lowercase"}
            path.write_text(json.dumps(configuration))
            with self.assertRaises(ValueError):
                ByteBPE(path)


if __name__ == "__main__":
    unittest.main()
