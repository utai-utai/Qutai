"""只读 mmap token 分片和可恢复批次；语言模型自己执行一次 next-token shift。"""

import hashlib
import json
import math
import mmap
import os
import random
import struct
import sys
import uuid
from array import array
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path


_TOKEN_TYPES = {"uint16": ("H", 2), "uint32": ("I", 4)}


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _tuples(value):
    return (
        tuple(_tuples(item) for item in value)
        if isinstance(value, (list, tuple))
        else value
    )


def write_token_shards(
    directory: str | Path,
    tokens,
    *,
    dtype: str = "uint16",
    shard_tokens: int = 1 << 20,
    tokenizer_hash: str | None = None,
) -> Path:
    """流式写入小端 token 文件；不覆盖已有 manifest 或正式语料。"""
    if dtype not in _TOKEN_TYPES or shard_tokens < 1:
        raise ValueError("分片需要 uint16/uint32 与正的 token 数")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"语料 manifest 已存在：{manifest}")
    code, width = _TOKEN_TYPES[dtype]
    generation = uuid.uuid4().hex
    records, created = [], []
    buffer = array(code)
    temporary = directory / f"manifest-{generation}.tmp"
    try:
        for token in tokens:
            if type(token) is not int or not 0 <= token < 1 << (width * 8):
                raise ValueError(f"token 编号超出 {dtype} 范围")
            buffer.append(token)
            if len(buffer) == shard_tokens:
                records.append(
                    _write_token_shard(directory, generation, records, buffer, created)
                )
                buffer = array(code)
        if buffer:
            records.append(
                _write_token_shard(directory, generation, records, buffer, created)
            )
        if not records:
            raise ValueError("语料至少需要一个 token")
        data = {
            "format": dtype + "-le",
            "tokenizer_sha256": tokenizer_hash,
            "tokens": sum(record["tokens"] for record in records),
            "shards": records,
        }
        with temporary.open("x", encoding="utf8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, manifest)
    except BaseException:
        for path in (*created, temporary):
            path.unlink(missing_ok=True)
        raise
    return manifest


def _write_token_shard(directory, generation, records, values, created):
    path = directory / f"tokens-{generation}-{len(records):05d}.bin"
    created.append(path)
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("xb") as file:
        values.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    return {"file": path.name, "tokens": len(values), "sha256": file_hash(path)}


def tokenize_shards(
    directory: str | Path,
    texts,
    tokenizer,
    *,
    shard_tokens: int = 1 << 20,
    dtype: str | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> Path:
    """逐文档编码；是否添加文档边界由调用方显式指定。"""
    dtype = (
        ("uint16" if tokenizer.size <= 65536 else "uint32") if dtype is None else dtype
    )
    return write_token_shards(
        directory,
        _document_tokens(texts, tokenizer, bos_token_id, eos_token_id),
        dtype=dtype,
        shard_tokens=shard_tokens,
        tokenizer_hash=tokenizer.sha256,
    )


def _document_tokens(texts, tokenizer, bos, eos):
    for text in texts:
        if bos is not None:
            yield bos
        yield from tokenizer.encode(text, add_special_tokens=False)
        if eos is not None:
            yield eos


class TokenShards:
    """把一个或多个 uint16/uint32 小端文件视为连续只读整数序列。"""

    def __init__(
        self,
        paths,
        *,
        dtype: str = "uint16",
        split: str | None = None,
        verify: bool = False,
        max_open: int = 8,
    ):
        self.manifest = None
        if isinstance(paths, (str, Path)):
            source = Path(paths)
            source = source / "manifest.json" if source.is_dir() else source
            if source.suffix == ".json":
                self.manifest = json.loads(source.read_text(encoding="utf8"))
                dtype = self.manifest.get("format", dtype).removesuffix("-le")
                records = self.manifest["shards"]
                if isinstance(records, dict):
                    if split is None:
                        raise ValueError("多划分语料需要指定 split")
                    records = records[split]
                self.paths = []
                for record in records:
                    relative = Path(record["file"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("语料分片路径必须位于 manifest 目录内")
                    self.paths.append(source.parent / relative)
                self.signature = file_hash(source) + (":" + split if split else "")
            else:
                self.paths, records = [source], None
        else:
            self.paths, records = [Path(path) for path in paths], None
        if dtype not in _TOKEN_TYPES or max_open < 1:
            raise ValueError("token 存储精度必须是 uint16/uint32，max_open 必须为正数")
        self.dtype, self._code, self._width = dtype, *_TOKEN_TYPES[dtype]
        self.max_open = max_open
        self._maps = OrderedDict()
        self.ends = []
        total = 0
        identity = []
        for index, path in enumerate(self.paths):
            info = path.stat()
            if info.st_size % self._width:
                raise ValueError(f"token 分片长度不匹配：{path}")
            count = info.st_size // self._width
            if records is not None:
                record = records[index]
                if count != record["tokens"] or (
                    verify and record.get("sha256") != file_hash(path)
                ):
                    raise ValueError(f"token 分片记录或哈希不匹配：{path}")
            total += count
            self.ends.append(total)
            identity.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
        if not total:
            raise ValueError("token 语料不能为空")
        if records is None:
            self.signature = hashlib.sha256(
                json.dumps([dtype, identity]).encode()
            ).hexdigest()
        self.tokenizer_hash = (
            self.manifest.get("tokenizer_sha256") if self.manifest else None
        )

    def __len__(self):
        return self.ends[-1]

    def _mapping(self, index):
        if index in self._maps:
            self._maps.move_to_end(index)
        else:
            if len(self._maps) == self.max_open:
                self._maps.popitem(last=False)[1].close()
            with self.paths[index].open("rb") as file:
                self._maps[index] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, end, step = index.indices(len(self))
            if step != 1:
                return [self[item] for item in range(start, end, step)]
            result = []
            while start < end:
                shard = bisect_right(self.ends, start)
                base = self.ends[shard - 1] if shard else 0
                take = min(end, self.ends[shard]) - start
                buffer = self._mapping(shard)[
                    (start - base) * self._width : (start - base + take) * self._width
                ]
                result.extend(
                    value[0] for value in struct.iter_unpack("<" + self._code, buffer)
                )
                start += take
            return result
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("token 索引超出范围")
        shard = bisect_right(self.ends, index)
        local = index - (self.ends[shard - 1] if shard else 0)
        return struct.unpack_from(
            "<" + self._code, self._mapping(shard), local * self._width
        )[0]

    def close(self):
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if hasattr(self, "_maps"):
            self.close()


class PackedTokens:
    """相邻窗口重叠一个边界 token；返回未 shift 的输入与独立标签列表。"""

    def __init__(self, tokens, sequence_length: int, *, split: str | None = None):
        if sequence_length < 2:
            raise ValueError("语言模型序列长度至少为 2")
        self.tokens = (
            tokens
            if isinstance(tokens, TokenShards)
            else TokenShards(tokens, split=split)
        )
        self.sequence_length = sequence_length
        self.signature = self.tokens.signature + f":sequence={sequence_length}"
        if not len(self):
            raise ValueError("语料 token 不足以组成一个完整序列")

    def __len__(self):
        return (len(self.tokens) - 1) // (self.sequence_length - 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("序列索引超出范围")
        start = index * (self.sequence_length - 1)
        values = self.tokens[start : start + self.sequence_length]
        return values, values.copy()


class StatefulBatcher:
    """O(1) 空间的无重复 epoch 排列，保存游标与随机状态即可精确接续。"""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int = 1,
        sequence_length: int | None = None,
        seed: int = 42,
        shuffle: bool = True,
        repeat: bool = False,
    ):
        if sequence_length is not None:
            dataset = PackedTokens(dataset, sequence_length)
        if not len(dataset) or batch_size < 1:
            raise ValueError("数据和 batch_size 必须非空且为正数")
        self.dataset, self.batch_size = dataset, batch_size
        self.seed, self.shuffle, self.repeat = seed, shuffle, repeat
        self.random = random.Random(seed)
        self.epoch, self.cursor = 0, 0
        self._permutation()

    def _permutation(self):
        size = len(self.dataset)
        self.multiplier, self.offset = 1, 0
        if self.shuffle and size > 1:
            self.multiplier = self.random.randrange(1, size)
            while math.gcd(self.multiplier, size) != 1:
                self.multiplier = (self.multiplier + 1) % size or 1
            self.offset = self.random.randrange(size)

    def next(self, batch_size: int | None = None):
        count = self.batch_size if batch_size is None else batch_size
        if count < 1:
            raise ValueError("batch_size 必须为正数")
        if not self.repeat and (
            self.epoch > 0 or count > len(self.dataset) - self.cursor
        ):
            raise StopIteration("语料已用尽；需要重复时显式设置 repeat=True")
        inputs, labels = [], []
        for _ in range(count):
            index = (self.multiplier * self.cursor + self.offset) % len(self.dataset)
            sample, target = self.dataset[index]
            inputs.append(sample)
            labels.append(target)
            self.cursor += 1
            if self.cursor == len(self.dataset):
                self.epoch += 1
                self.cursor = 0
                self._permutation()
        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "repeat": self.repeat,
            "batch_size": self.batch_size,
            "length": len(self.dataset),
            "signature": self.dataset.signature,
            "multiplier": self.multiplier,
            "offset": self.offset,
            "random_state": self.random.getstate(),
        }

    def validate_state_dict(self, state):
        size = len(self.dataset)
        if state["length"] != size or state["signature"] != self.dataset.signature:
            raise ValueError("恢复语料或序列长度与 checkpoint 不一致")
        if (
            type(state["epoch"]) is not int
            or state["epoch"] < 0
            or type(state["cursor"]) is not int
            or not 0 <= state["cursor"] < size
            or math.gcd(state["multiplier"], size) != 1
            or not 0 <= state["offset"] < size
            or state["batch_size"] < 1
        ):
            raise ValueError("checkpoint 中的数据游标或排列无效")
        restored = random.Random()
        restored.setstate(_tuples(state["random_state"]))
        return restored

    def load_state_dict(self, state):
        restored = self.validate_state_dict(state)
        for name in (
            "epoch",
            "cursor",
            "seed",
            "shuffle",
            "repeat",
            "batch_size",
            "multiplier",
            "offset",
        ):
            setattr(self, name, state[name])
        self.random = restored
