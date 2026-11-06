"""流式二进制训练 checkpoint：先完整写分片，最后原子发布 manifest。"""

import hashlib
import json
import os
import random
import sys
import uuid
from array import array
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ..backends.device import get_backend
from ..core.tensor import Tensor
from .data import _tuples, file_hash


_WIDTHS = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16, "int64": 8}
_CODES = {
    "float32": "f",
    "float64": "d",
    "complex64": "f",
    "complex128": "d",
    "int64": "q",
}


def _manifest_path(path):
    path = Path(path)
    return path if path.suffix == ".json" else path / "manifest.json"


def checkpoint_metadata(path: str | Path, *, verify: bool = False) -> dict:
    """读取模型配置等小元数据；verify=True 额外流式核对二进制分片。"""
    manifest = _manifest_path(path)
    metadata = json.loads(manifest.read_text(encoding="utf8"))
    if verify:
        _validate_files(manifest, metadata)
    return metadata


def _named_parameters(model):
    names = (
        list(model.named_parameters())
        if hasattr(model, "named_parameters")
        else [
            (f"parameters.{index}", parameter)
            for index, parameter in enumerate(model.parameters())
        ]
    )
    if len({name for name, _ in names}) != len(names):
        raise ValueError("参数名称不能重复")
    return names


def _model_buffers(model):
    if hasattr(model, "named_buffers"):
        return dict(model.named_buffers())
    result, visited = {}, set()
    pending = [("", model)]
    while pending:
        prefix, layer = pending.pop()
        if id(layer) in visited:
            continue
        visited.add(id(layer))
        for name, storage in (
            layer.buffers().items() if hasattr(layer, "buffers") else ()
        ):
            result[prefix + name] = storage
        for name, value in getattr(layer, "__dict__", {}).items():
            children = (
                enumerate(value)
                if isinstance(value, (list, tuple))
                else ((None, value),)
            )
            for index, child in children:
                if not isinstance(child, Tensor) and hasattr(child, "parameters"):
                    pending.append(
                        (
                            prefix
                            + name
                            + (f".{index}" if index is not None else "")
                            + ".",
                            child,
                        )
                    )
    return result


def _allocate_storage(device, record):
    backend = get_backend(device)
    if device == "cuda":
        return backend.CUDAStorage(record["size"], record["dtype"])
    if device == "cpu":
        result = backend.storage((), dtype=record["dtype"])
        result.buffer = array(_CODES[record["dtype"]], [0]) * (
            record["nbytes"] // array(_CODES[record["dtype"]]).itemsize
        )
        return result
    raise ValueError("二进制 checkpoint 目前支持 CPU 与 CUDA 存储")


def _tokenizer_hash(tokenizer):
    if tokenizer is None:
        return None
    if isinstance(tokenizer, (str, Path)):
        return file_hash(tokenizer)
    if hasattr(tokenizer, "sha256"):
        return tokenizer.sha256
    if hasattr(tokenizer, "characters"):
        return hashlib.sha256(tokenizer.characters.encode("utf8")).hexdigest()
    raise TypeError("tokenizer 需要提供 sha256 或文件路径")


def _config(model):
    config = getattr(model, "config", None)
    return asdict(config) if is_dataclass(config) else config


def _storage_chunks(storage, chunk_bytes):
    width = _WIDTHS[storage.dtype]
    chunk_bytes = max(width, chunk_bytes // width * width)
    if storage.device == "cpu":
        host = memoryview(storage.buffer).cast("B")
        for offset in range(0, storage.nbytes, chunk_bytes):
            piece = host[offset : offset + chunk_bytes]
            if sys.byteorder == "little":
                yield piece
            else:
                values = array(_CODES[storage.dtype])
                values.frombytes(piece)
                values.byteswap()
                yield memoryview(values).cast("B")
    elif storage.device == "cuda":
        storage.runtime.synchronize()
        for offset in range(0, storage.nbytes, chunk_bytes):
            host = bytearray(min(chunk_bytes, storage.nbytes - offset))
            storage.runtime.download(storage.ptr + offset, host)
            yield host
    else:
        raise ValueError("二进制 checkpoint 目前支持 CPU 与 CUDA 存储")


class _ShardWriter:
    def __init__(self, directory, shard_bytes, chunk_bytes):
        self.directory, self.shard_bytes, self.chunk_bytes = (
            directory,
            shard_bytes,
            chunk_bytes,
        )
        self.generation = uuid.uuid4().hex
        self.shards, self.created = [], []
        self.file = None
        self.size = 0

    def _open(self):
        self.name = f"data-{self.generation}-{len(self.shards):05d}.bin"
        self.temporary = self.directory / (self.name + ".part")
        self.created.extend((self.temporary, self.directory / self.name))
        self.file = self.temporary.open("xb")
        self.size, self.digest = 0, hashlib.sha256()

    def finish(self):
        if self.file is None:
            return
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.file = None
        os.replace(self.temporary, self.directory / self.name)
        self.shards.append(
            {"file": self.name, "nbytes": self.size, "sha256": self.digest.hexdigest()}
        )

    def write(self, storage):
        segments = []
        for chunk in _storage_chunks(storage, self.chunk_bytes):
            remaining = memoryview(chunk).cast("B")
            while remaining:
                if self.file is None:
                    self._open()
                take = min(len(remaining), self.shard_bytes - self.size)
                part = remaining[:take]
                segments.append(
                    {"file": self.name, "offset": self.size, "nbytes": take}
                )
                self.file.write(part)
                self.digest.update(part)
                self.size += take
                remaining = remaining[take:]
                if self.size == self.shard_bytes:
                    self.finish()
        return {
            "dtype": storage.dtype,
            "size": storage.size,
            "nbytes": storage.nbytes,
            "segments": segments,
        }

    def discard(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        for path in self.created:
            path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    global_step: int = 0,
    tokens: int = 0,
    learning_rate: float | None = None,
    data=None,
    tokenizer=None,
    rngs=None,
    state: dict | None = None,
    shard_bytes: int = 256 << 20,
    chunk_bytes: int = 4 << 20,
) -> Path:
    """保存参数、梯度、优化器、随机数与数据位置；只占一个小块主机缓冲区。"""
    if shard_bytes < 1 or chunk_bytes < 1 or global_step < 0 or tokens < 0:
        raise ValueError("分片和缓冲区必须为正数，训练计数不能为负数")
    manifest = _manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    previous = checkpoint_metadata(manifest) if manifest.exists() else None
    writer = _ShardWriter(manifest.parent, shard_bytes, chunk_bytes)
    temporary = manifest.parent / f"manifest-{writer.generation}.tmp"
    published = False
    try:
        named = _named_parameters(model)
        unique, aliases, parameters, tensors = {}, {}, [], {}
        for name, parameter in named:
            if parameter not in unique:
                reference = f"parameter.{len(unique)}"
                unique[parameter] = reference
                tensors[reference] = writer.write(parameter.storage)
                gradient = None
                if parameter.grad is not None:
                    gradient = reference + ".gradient"
                    tensors[gradient] = writer.write(parameter.grad.storage)
                parameters.append(
                    {
                        "name": name,
                        "tensor": reference,
                        "shape": parameter.shape,
                        "requires_grad": parameter.requires_grad,
                        "gradient": gradient,
                    }
                )
            aliases[name] = unique[parameter]
        buffers = {}
        for name, storage in _model_buffers(model).items():
            reference = f"buffer.{len(buffers)}"
            buffers[name] = reference
            tensors[reference] = writer.write(storage)
        optimizer_record = None
        if optimizer is not None:
            if any(parameter not in unique for parameter in optimizer.parameters):
                raise ValueError("优化器包含模型之外的参数")
            optimizer_record = {
                "type": type(optimizer).__name__,
                "parameters": [unique[parameter] for parameter in optimizer.parameters],
                "settings": {
                    name: getattr(optimizer, name) for name in optimizer._settings
                },
                "buffers": {},
                "steps": {
                    unique[parameter]: value
                    for parameter, value in getattr(optimizer, "steps", {}).items()
                },
            }
            for name in optimizer._buffers:
                records = {}
                for parameter, storage in getattr(optimizer, name).items():
                    reference = f"optimizer.{name}.{unique[parameter]}"
                    tensors[reference] = writer.write(storage)
                    records[unique[parameter]] = reference
                optimizer_record["buffers"][name] = records
            if hasattr(optimizer, "param_groups"):
                optimizer_record["groups"] = [
                    {
                        "parameters": [
                            unique[parameter]
                            for parameter in group.get(
                                "params", group.get("parameters", ())
                            )
                        ],
                        "settings": {
                            name: value
                            for name, value in group.items()
                            if name not in ("params", "parameters")
                        },
                    }
                    for group in optimizer.param_groups
                ]
        writer.finish()
        metadata = {
            "format": "qutai-training",
            "version": 1,
            "model_type": type(model).__name__,
            "model_config": _config(model),
            "parameters": parameters,
            "aliases": aliases,
            "tensors": tensors,
            "buffers": buffers,
            "optimizer": optimizer_record,
            "shards": writer.shards,
            "tokenizer_sha256": _tokenizer_hash(tokenizer),
            "training": {
                "global_step": global_step,
                "tokens": tokens,
                "learning_rate": learning_rate,
                "data": data.state_dict() if data is not None else None,
                "extra": state or {},
            },
            "rng": {
                "python": random.getstate(),
                "named": {name: rng.getstate() for name, rng in (rngs or {}).items()},
            },
        }
        if hasattr(model, "get_rng_state"):
            metadata["rng"]["model"] = model.get_rng_state()
        with temporary.open("x", encoding="utf8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, manifest)
        published = True
        # 发布后才清理上一代，写入或编译中途失败不会破坏此前 checkpoint。
        for shard in previous.get("shards", []) if previous else ():
            old = Path(shard["file"])
            if old.name == str(old) and old.name.startswith("data-"):
                try:
                    (manifest.parent / old).unlink(missing_ok=True)
                except OSError:
                    pass
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        if not published:
            writer.discard()


def _validate_files(manifest, metadata):
    if metadata.get("format") != "qutai-training" or metadata.get("version") != 1:
        raise ValueError("不是支持的 qutai 训练 checkpoint")
    shards = {}
    for shard in metadata["shards"]:
        relative = Path(shard["file"])
        if relative.name != str(relative) or relative.is_absolute():
            raise ValueError("checkpoint 分片路径必须是同目录文件名")
        path = manifest.parent / relative
        if path.stat().st_size != shard["nbytes"] or file_hash(path) != shard["sha256"]:
            raise ValueError(f"checkpoint 分片大小或哈希不匹配：{relative}")
        shards[shard["file"]] = shard["nbytes"]
    for record in metadata["tensors"].values():
        if (
            record["dtype"] not in _WIDTHS
            or record["size"] < 0
            or record["nbytes"] != record["size"] * _WIDTHS[record["dtype"]]
        ):
            raise ValueError("checkpoint 张量精度或大小无效")
        if sum(segment["nbytes"] for segment in record["segments"]) != record["nbytes"]:
            raise ValueError("checkpoint 张量分片长度不匹配")
        for segment in record["segments"]:
            if (
                segment["file"] not in shards
                or segment["offset"] < 0
                or segment["nbytes"] < 0
                or segment["offset"] + segment["nbytes"] > shards[segment["file"]]
            ):
                raise ValueError("checkpoint 张量分片越界")


def _read_storage(directory, record, storage, chunk_bytes):
    if storage.dtype != record["dtype"] or storage.size != record["size"]:
        raise ValueError("目标存储的精度或元素数量与 checkpoint 不同")
    destination = (
        memoryview(storage.buffer).cast("B") if storage.device == "cpu" else None
    )
    if storage.device not in ("cpu", "cuda"):
        raise ValueError("二进制 checkpoint 目前支持 CPU 与 CUDA 存储")
    offset = 0
    for segment in record["segments"]:
        with (directory / segment["file"]).open("rb") as file:
            file.seek(segment["offset"])
            remaining = segment["nbytes"]
            while remaining:
                content = bytearray(file.read(min(chunk_bytes, remaining)))
                if not content:
                    raise ValueError("checkpoint 数据被截断")
                if storage.device == "cpu":
                    destination[offset : offset + len(content)] = content
                else:
                    storage.runtime.upload(storage.ptr + offset, content)
                offset += len(content)
                remaining -= len(content)
    if storage.device == "cpu" and sys.byteorder != "little":
        storage.buffer.byteswap()


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    data=None,
    tokenizer=None,
    rngs=None,
    restore_rng: bool = True,
    strict: bool = True,
    chunk_bytes: int = 4 << 20,
) -> dict:
    """恢复到既有模型，保留参数 Tensor/Storage 身份；先校验全部文件再修改内存。"""
    if chunk_bytes < 1:
        raise ValueError("checkpoint 读取缓冲区必须为正数")
    manifest = _manifest_path(path)
    metadata = checkpoint_metadata(manifest)
    _validate_files(manifest, metadata)
    if tokenizer is not None and metadata["tokenizer_sha256"] != _tokenizer_hash(
        tokenizer
    ):
        raise ValueError("checkpoint 的 tokenizer 哈希不一致")
    # JSON 统一把 tuple 规范成 list 后比较配置。
    if strict and metadata["model_config"] != json.loads(json.dumps(_config(model))):
        raise ValueError("checkpoint 模型配置与目标模型不一致")
    named = dict(_named_parameters(model))
    if set(named) != set(metadata["aliases"]):
        raise ValueError("checkpoint 参数名称与目标模型不一致")
    by_reference = {}
    for name, reference in metadata["aliases"].items():
        if reference in by_reference and by_reference[reference] is not named[name]:
            raise ValueError("目标模型没有保留 checkpoint 的共享参数关系")
        by_reference[reference] = named[name]
    if len(set(by_reference.values())) != len(by_reference):
        raise ValueError("目标模型的共享参数关系与 checkpoint 不一致")
    tensors = metadata["tensors"]
    for record in metadata["parameters"]:
        parameter = by_reference[record["tensor"]]
        saved = tensors[record["tensor"]]
        if (
            list(parameter.shape) != record["shape"]
            or parameter.dtype != saved["dtype"]
            or parameter.size != saved["size"]
        ):
            raise ValueError("checkpoint 参数形状或精度不一致")
        gradient = (
            tensors[record["gradient"]] if record["gradient"] is not None else None
        )
        if gradient is not None and (gradient["size"], gradient["dtype"]) != (
            parameter.size,
            parameter.dtype,
        ):
            raise ValueError("checkpoint 梯度精度或元素数量不一致")
    buffers = _model_buffers(model)
    if set(buffers) != set(metadata.get("buffers", {})):
        raise ValueError("checkpoint 模型运行缓冲区不一致")
    for name, reference in metadata.get("buffers", {}).items():
        saved = tensors[reference]
        if buffers[name].size != saved["size"] or buffers[name].dtype != saved["dtype"]:
            raise ValueError("checkpoint 模型缓冲区精度或大小不一致")
    saved_optimizer = metadata["optimizer"]
    if optimizer is not None:
        if (
            saved_optimizer is None
            or type(optimizer).__name__ != saved_optimizer["type"]
        ):
            raise ValueError("checkpoint 优化器类型不一致")
        if list(optimizer.parameters) != [
            by_reference[reference] for reference in saved_optimizer["parameters"]
        ]:
            raise ValueError("checkpoint 优化器参数顺序不一致")
        if set(saved_optimizer["buffers"]) != set(optimizer._buffers):
            raise ValueError("checkpoint 优化器状态字段不一致")
        for records in saved_optimizer["buffers"].values():
            for reference, storage_reference in records.items():
                if (
                    reference not in by_reference
                    or tensors[storage_reference]["size"]
                    != by_reference[reference].size
                ):
                    raise ValueError("checkpoint 优化器状态大小不一致")
        if any(
            reference not in by_reference or type(step) is not int or step < 1
            for reference, step in saved_optimizer["steps"].items()
        ):
            raise ValueError("checkpoint 优化器更新次数无效")
        if "groups" in saved_optimizer:
            if [
                reference
                for group in saved_optimizer["groups"]
                for reference in group["parameters"]
            ] != saved_optimizer["parameters"]:
                raise ValueError("checkpoint 优化器参数组顺序不一致")
        # 让优化器自己的构造函数校验超参数；不产生参数或状态副本。
        settings = saved_optimizer["settings"]
        groups = [
            {
                **group["settings"],
                "params": [
                    by_reference[reference] for reference in group["parameters"]
                ],
            }
            for group in saved_optimizer.get("groups", [])
        ]
        type(optimizer)(groups or optimizer.parameters, **settings)
    if restore_rng and set(metadata["rng"]["named"]) != set(rngs or {}):
        raise ValueError("checkpoint 命名随机数生成器不一致")
    if restore_rng:
        for saved in (metadata["rng"]["python"], *metadata["rng"]["named"].values()):
            random.Random().setstate(_tuples(saved))
        if "model" in metadata["rng"] and not hasattr(model, "set_rng_state"):
            raise ValueError("目标模型不能恢复 checkpoint 随机状态")
    if (
        data is not None
        and metadata["training"]["data"] is not None
        and hasattr(data, "validate_state_dict")
    ):
        data.validate_state_dict(metadata["training"]["data"])
    for record in metadata["parameters"]:
        parameter = by_reference[record["tensor"]]
        _read_storage(
            manifest.parent, tensors[record["tensor"]], parameter.storage, chunk_bytes
        )
        parameter.requires_grad = record["requires_grad"]
        gradient = record["gradient"]
        if gradient is None:
            parameter.grad = None
        else:
            saved = tensors[gradient]
            if (
                parameter.grad is not None
                and parameter.grad.dtype == saved["dtype"]
                and parameter.grad.device == parameter.device
            ):
                storage = parameter.grad.storage
            else:
                storage = _allocate_storage(parameter.device, saved)
                parameter.grad = Tensor(storage, shape=parameter.shape)
            _read_storage(manifest.parent, saved, storage, chunk_bytes)
    for name, reference in metadata.get("buffers", {}).items():
        _read_storage(manifest.parent, tensors[reference], buffers[name], chunk_bytes)
    if optimizer is not None:
        for name, value in saved_optimizer["settings"].items():
            setattr(optimizer, name, tuple(value) if name == "betas" else value)
        for name, records in saved_optimizer["buffers"].items():
            restored = {}
            for reference, storage_reference in records.items():
                saved = tensors[storage_reference]
                storage = getattr(optimizer, name).get(by_reference[reference])
                if storage is None or (storage.size, storage.dtype, storage.device) != (
                    saved["size"],
                    saved["dtype"],
                    by_reference[reference].device,
                ):
                    storage = _allocate_storage(by_reference[reference].device, saved)
                _read_storage(manifest.parent, saved, storage, chunk_bytes)
                restored[by_reference[reference]] = storage
            setattr(optimizer, name, restored)
        if hasattr(optimizer, "steps"):
            optimizer.steps = {
                by_reference[reference]: step
                for reference, step in saved_optimizer["steps"].items()
            }
        if "groups" in saved_optimizer and hasattr(optimizer, "param_groups"):
            optimizer.param_groups = [
                {
                    **group["settings"],
                    "params": tuple(
                        by_reference[reference] for reference in group["parameters"]
                    ),
                }
                for group in saved_optimizer["groups"]
            ]
    training = metadata["training"]
    if data is not None and training["data"] is not None:
        data.load_state_dict(training["data"])
    if restore_rng:
        random.setstate(_tuples(metadata["rng"]["python"]))
        for name, state in metadata["rng"]["named"].items():
            rngs[name].setstate(_tuples(state))
        if "model" in metadata["rng"]:
            model.set_rng_state(metadata["rng"]["model"])
    return {
        **training,
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "model_config": metadata["model_config"],
    }
