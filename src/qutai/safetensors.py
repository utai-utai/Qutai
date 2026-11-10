from array import array
from collections.abc import Mapping
import json
from math import prod
import mmap
from pathlib import Path
import struct
import sys

from .backends import cuda
from .backends.device import get_backend
from .core.tensor import Tensor

_DTYPES = {'F32': ('float32', 'f', 4), 'F64': ('float64', 'd', 8), 'BF16': ('bfloat16', 'H', 2)}


class SafeTensors(Mapping):
    def __init__(self, path, device='cpu'):
        self.path, self.device, self._cache = Path(path), device, {}
        get_backend(device)
        with self.path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8: raise ValueError('无效 safetensors 文件')
            size = struct.unpack('<Q', raw)[0]
            if size > 16 * 1024 * 1024 or size > self.path.stat().st_size - 8:
                raise ValueError('无效 safetensors 头部长度')
            header = json.loads(stream.read(size))
        if not isinstance(header, dict): raise ValueError('safetensors 头部必须为字典')
        self.metadata = header.pop('__metadata__', {})
        if not isinstance(self.metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in self.metadata.items()):
            raise ValueError('safetensors 元数据必须为字符串字典')
        self._header, self._start = header, size + 8
        spans = []
        for name, item in header.items():
            if not isinstance(item, dict) or item.get('dtype') not in _DTYPES:
                raise ValueError(f'不支持的 safetensors 张量：{name}')
            shape, offsets = item.get('shape'), item.get('data_offsets')
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f'无效张量形状：{name}')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
                raise ValueError(f'无效张量偏移：{name}')
            start, end = offsets
            if not 0 <= start <= end <= self.path.stat().st_size - self._start or end-start != prod(shape)*_DTYPES[item['dtype']][2]:
                raise ValueError(f'无效张量数据长度：{name}')
            if start != end: spans.append((start, end))
        spans.sort()
        if any(a[1] > b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError('safetensors 张量数据重叠')

    def __iter__(self):
        return iter(self._header)

    def __len__(self):
        return len(self._header)

    def __contains__(self, name):
        return name in self._header

    def __getitem__(self, name):
        if name not in self._cache:
            item = self._header[name]
            dtype, code, width = _DTYPES[item['dtype']]
            if dtype == 'bfloat16' and self.device != 'cuda':
                raise ValueError('BF16 safetensors 当前需要 CUDA 设备')
            start, end = item['data_offsets']
            with self.path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as data:
                with memoryview(data)[self._start+start:self._start+end] as view:
                    if self.device == 'cuda':
                        storage = cuda.CUDAStorage(prod(item['shape']), dtype)
                        storage.runtime.upload(storage.ptr, view)
                    else:
                        values = array(code)
                        values.frombytes(view)
                        if sys.byteorder != 'little': values.byteswap()
                        storage = get_backend(self.device).storage(values, dtype=dtype)
            self._cache[name] = Tensor(storage, shape=tuple(item['shape']))
        return self._cache[name]


def save_safetensors(path, tensors, *, metadata=None):
    tensors = dict(tensors)
    header, offset = {}, 0
    for name, tensor in tensors.items():
        if not isinstance(name, str) or name == '__metadata__': raise ValueError('无效张量名称')
        kind = next((key for key, value in _DTYPES.items() if value[0] == tensor.dtype), None)
        if kind is None: raise ValueError(f'不支持的张量精度：{tensor.dtype}')
        size = tensor.size * _DTYPES[kind][2]
        header[name] = dict(dtype=kind, shape=list(tensor.shape), data_offsets=[offset, offset+size])
        offset += size
    if metadata is not None:
        if not isinstance(metadata, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in metadata.items()):
            raise ValueError('metadata 必须为字符串字典')
        header['__metadata__'] = metadata
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for tensor in tensors.values():
            if tensor.device == 'cuda':
                values = bytearray(tensor.storage.nbytes)
                tensor.storage.runtime.download(tensor.storage.ptr, values)
                stream.write(values)
            else:
                code = 'f' if tensor.dtype == 'float32' else 'd'
                values = array(code, tensor.storage.tolist())
                if sys.byteorder != 'little': values.byteswap()
                values.tofile(stream)
    temporary.replace(path)
