"""读取 ByteLevel BPE tokenizer.json；编码只使用 Python 标准库。"""

import hashlib
import heapq
import json
import re
import unicodedata
from pathlib import Path


SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|mode_direct|>",
    "<|mode_think|>",
    "<think>",
    "</think>",
)


def _byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = values.copy()
    extra = 256
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


_BYTE_CHAR = _byte_alphabet()
_CHAR_BYTE = {character: value for value, character in _BYTE_CHAR.items()}
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _kind(character):
    category = unicodedata.category(character)[0]
    if category in ("L", "N"):
        return category
    # Unicode White_Space；str.isspace 还包含四个 ASCII 信息分隔符。
    return (
        "space"
        if character.isspace() and character not in "\x1c\x1d\x1e\x1f"
        else "other"
    )


def _pieces(text):
    """逐项实现 GPT-2 的 Unicode 分类正则，不依赖第三方 regex。"""
    index = 0
    while index < len(text):
        contraction = next(
            (part for part in _CONTRACTIONS if text.startswith(part, index)), None
        )
        if contraction:
            yield contraction
            index += len(contraction)
            continue
        start = index
        if (
            text[index] == " "
            and index + 1 < len(text)
            and _kind(text[index + 1]) != "space"
        ):
            index += 1
        kind = _kind(text[index])
        index += 1
        while index < len(text) and _kind(text[index]) == kind:
            index += 1
        if kind == "space" and index < len(text) and index - start > 1:
            index -= 1
        yield text[start:index]


class ByteBPE:
    """兼容 UtaiLM 的固定 ByteLevel BPE 资产，不训练或修改 tokenizer 文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        configuration = json.loads(content)
        model = configuration["model"]
        pretokenizer = configuration.get("pre_tokenizer") or {}
        decoder = configuration.get("decoder") or {}
        if (
            model.get("type") != "BPE"
            or pretokenizer.get("type") != "ByteLevel"
            or decoder.get("type") != "ByteLevel"
            or configuration.get("normalizer") is not None
            or configuration.get("post_processor") is not None
            or configuration.get("truncation") is not None
            or configuration.get("padding") is not None
            or model.get("dropout") not in (None, 0)
            or model.get("byte_fallback")
            or model.get("continuing_subword_prefix")
            or model.get("end_of_word_suffix")
        ):
            raise ValueError("只支持无额外归一化、截断和后处理的 ByteLevel BPE 配置")
        self.vocab = dict(model["vocab"])
        self._tokens = {value: key for key, value in self.vocab.items()}
        if len(self._tokens) != len(self.vocab) or set(self._tokens) != set(
            range(len(self.vocab))
        ):
            raise ValueError("BPE 词表编号必须连续且不重复")
        if any(character not in self.vocab for character in _CHAR_BYTE):
            raise ValueError("ByteLevel BPE 词表缺少字节字母表")
        self._merges = {}
        for rank, pair in enumerate(model["merges"]):
            left, right = pair.split(" ") if isinstance(pair, str) else pair
            self._merges[(self.vocab[left], self.vocab[right])] = (
                rank,
                self.vocab[left + right],
            )
        self.add_prefix_space = pretokenizer.get("add_prefix_space", True)
        self.use_regex = pretokenizer.get("use_regex", True)
        self.ignore_merges = model.get("ignore_merges", False)
        added = configuration.get("added_tokens", [])
        if any(
            any(
                token.get(flag, False)
                for flag in ("single_word", "lstrip", "rstrip", "normalized")
            )
            for token in added
        ):
            raise ValueError("此 ByteBPE 暂不支持带匹配修饰符的 added_tokens")
        self._added = {token["content"]: token["id"] for token in added}
        self._special_ids = {
            token["id"] for token in added if token.get("special", False)
        }
        self._special_pattern = (
            re.compile(
                "|".join(
                    re.escape(token)
                    for token in sorted(self._added, key=len, reverse=True)
                )
            )
            if added
            else None
        )
        self._cache = {}

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(path)

    @property
    def size(self) -> int:
        return len(self.vocab)

    @property
    def hash(self) -> str:
        return self.sha256

    def get_vocab_size(self) -> int:
        return self.size

    def token_to_id(self, token: str) -> int | None:
        return self._added.get(token, self.vocab.get(token))

    def id_to_token(self, index: int) -> str | None:
        return self._tokens.get(index)

    def _merge(self, piece: str) -> tuple[int, ...]:
        if piece in self._cache:
            return self._cache[piece]
        encoded = "".join(_BYTE_CHAR[value] for value in piece.encode("utf8"))
        if self.ignore_merges and encoded in self.vocab:
            return (self.vocab[encoded],)
        symbols = [self.vocab[character] for character in encoded]
        previous = list(range(-1, len(symbols) - 1))
        following = list(range(1, len(symbols))) + [-1]
        alive = [True] * len(symbols)
        queue = []
        for left in range(len(symbols) - 1):
            pair = (symbols[left], symbols[left + 1])
            if pair in self._merges:
                rank, merged = self._merges[pair]
                heapq.heappush(queue, (rank, left, left + 1, *pair, merged))
        while queue:
            rank, left, right, left_id, right_id, merged = heapq.heappop(queue)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or (symbols[left], symbols[right]) != (left_id, right_id)
            ):
                continue
            symbols[left], alive[right] = merged, False
            following[left] = following[right]
            if following[right] != -1:
                previous[following[right]] = left
            for candidate in (previous[left], left):
                if candidate == -1 or following[candidate] == -1:
                    continue
                neighbor = following[candidate]
                pair = (symbols[candidate], symbols[neighbor])
                if pair in self._merges:
                    rank, merged = self._merges[pair]
                    heapq.heappush(queue, (rank, candidate, neighbor, *pair, merged))
        result = tuple(symbol for symbol, valid in zip(symbols, alive) if valid)
        if len(self._cache) >= 16384:
            self._cache.pop(next(iter(self._cache)))
        self._cache[piece] = result
        return result

    def _encode_text(self, text):
        if not text:
            return []
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        return [
            index
            for piece in (_pieces(text) if self.use_regex else (text,))
            for index in self._merge(piece)
        ]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """返回编号；资产无 post_processor，因此不会自动添加 BOS/EOS。"""
        if self._special_pattern is None:
            return self._encode_text(text)
        result, start = [], 0
        for match in self._special_pattern.finditer(text):
            result.extend(self._encode_text(text[start : match.start()]))
            result.append(self._added[match.group()])
            start = match.end()
        result.extend(self._encode_text(text[start:]))
        return result

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        """按字节连接后统一 UTF-8 解码，保留跨 token 的多字节字符。"""
        content = bytearray()
        for index in ids:
            if type(index) is not int or index not in self._tokens:
                raise ValueError("BPE 编号必须是词表范围内的非负整数")
            if skip_special_tokens and index in self._special_ids:
                continue
            token = self._tokens[index]
            content.extend(
                bytes(_CHAR_BYTE[character] for character in token)
                if all(character in _CHAR_BYTE for character in token)
                else token.encode("utf8")
            )
        return content.decode("utf8", "replace")
