"""标准库 ctypes 封装的单流 CUDA Driver / NVRTC 运行时；导入时不加载原生库。"""

import atexit
import ctypes
import ctypes.util
import hashlib
import os
import sys
import threading
from pathlib import Path


_runtime = None
_runtime_lock = threading.Lock()


def _toolkit_roots() -> list[Path]:
    roots = [Path(value) for key, value in os.environ.items()
             if value and (key in ("CUDA_PATH", "CUDA_HOME") or key.startswith("CUDA_PATH_V"))]
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
        roots.extend(sorted(base.glob("v*"), reverse=True))
    else:
        roots.extend((Path("/usr/local/cuda"), Path("/opt/cuda")))
        roots.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
    return list(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))


def _libraries():
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("CUDA 后端需要 64 位 Python")
    roots = _toolkit_roots()
    directories = []
    candidates = []
    if sys.platform == "win32":
        loader = ctypes.WinDLL
        for root in roots:
            for directory in (root / "bin/x64", root / "bin"):
                if directory.is_dir():
                    directories.append(os.add_dll_directory(str(directory)))
                    candidates.extend(sorted(directory.glob("nvrtc64_*.dll"), reverse=True))
        driver_name = "nvcuda.dll"
        candidates.append("nvrtc64_130_0.dll")
    elif sys.platform.startswith("linux"):
        loader = ctypes.CDLL
        driver_name = "libcuda.so.1"
        for root in roots:
            for directory in (root / "lib64", root / "targets/x86_64-linux/lib"):
                candidates.extend(sorted(directory.glob("libnvrtc.so*"), reverse=True))
        candidates.extend(filter(None, (ctypes.util.find_library("nvrtc"), "libnvrtc.so")))
    else:
        raise RuntimeError("CUDA 后端只支持 Windows 和 Linux")
    try:
        driver = loader(driver_name)
        errors = []
        for candidate in candidates:
            try:
                compiler = loader(str(candidate))
                break
            except OSError as error:
                errors.append(str(error))
        else:
            raise RuntimeError("未找到可用的 NVRTC，请安装 CUDA Toolkit 并配置 CUDA_PATH：" + "；".join(errors))
        includes = [root / "include" for root in roots if (root / "include").is_dir()]
        return driver, compiler, includes, directories
    except (OSError, RuntimeError) as error:
        for directory in directories:
            directory.close()
        raise RuntimeError(f"CUDA 原生运行时加载失败：{error}") from error


def _bind(library, name, arguments, result=ctypes.c_int, fallback=None):
    function = getattr(library, name, None)
    if function is None and fallback:
        function = getattr(library, fallback, None)
    if function is None:
        raise RuntimeError(f"CUDA 原生运行时缺少接口：{name}")
    function.argtypes = arguments
    function.restype = result
    return function


class Runtime:
    """单 GPU 默认流；内核异步排队，显式读回时同步。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._context = ctypes.c_void_p()
        self._device = ctypes.c_int()
        self._allocations = {}
        self._blocks = {}
        self._available = {}
        self._sources = {}
        self.allocated_bytes = self.reserved_bytes = 0
        self.peak_allocated_bytes = self.peak_reserved_bytes = 0
        self.cache_limit_bytes = 1 << 30
        self.tensor_core_launches = 0
        self.debug_sync = os.environ.get("QUTAI_CUDA_SYNC") == "1"
        self._modules = {}
        self._functions = {}
        self._closed = False
        self.launches = 0
        self._driver, self._compiler, self._includes, self._dll_directories = _libraries()
        try:
            self._bind_apis()
            self._check(self._init(0), "初始化驱动")
            self._check(self._device_get(ctypes.byref(self._device), 0), "选择第一个 GPU")
            self._check(self._retain(ctypes.byref(self._context), self._device), "取得 GPU 上下文")
            self._current()
            name = ctypes.create_string_buffer(256)
            self._check(self._device_name(name, len(name), self._device), "读取 GPU 名称")
            self.name = name.value.decode("utf8", "replace")
            attributes = []
            for attribute in (75, 76):
                value = ctypes.c_int()
                self._check(self._device_attribute(ctypes.byref(value), attribute, self._device), "读取 GPU 计算能力")
                attributes.append(value.value)
            self.compute_capability = tuple(attributes)
            version = ctypes.c_int()
            self._check(self._driver_version(ctypes.byref(version)), "读取驱动版本")
            self.driver_version = version.value
            major, minor = ctypes.c_int(), ctypes.c_int()
            self._nvcheck(self._nv_version(ctypes.byref(major), ctypes.byref(minor)), "读取 NVRTC 版本")
            self.nvrtc_version = (major.value, minor.value)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _bind_apis(self):
        pointer, integer, unsigned = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        size, device_pointer = ctypes.c_size_t, ctypes.c_uint64
        address = ctypes.POINTER(pointer)
        strings = ctypes.POINTER(ctypes.c_char_p)
        driver = self._driver
        self._init = _bind(driver, "cuInit", [unsigned])
        self._device_get = _bind(driver, "cuDeviceGet", [ctypes.POINTER(integer), integer])
        self._device_name = _bind(driver, "cuDeviceGetName", [pointer, integer, integer])
        self._device_attribute = _bind(driver, "cuDeviceGetAttribute", [ctypes.POINTER(integer), integer, integer])
        self._driver_version = _bind(driver, "cuDriverGetVersion", [ctypes.POINTER(integer)])
        self._retain = _bind(driver, "cuDevicePrimaryCtxRetain", [address, integer])
        self._release = _bind(driver, "cuDevicePrimaryCtxRelease_v2", [integer], fallback="cuDevicePrimaryCtxRelease")
        self._set_current = _bind(driver, "cuCtxSetCurrent", [pointer])
        self._sync = _bind(driver, "cuCtxSynchronize", [])
        self._malloc = _bind(driver, "cuMemAlloc_v2", [ctypes.POINTER(device_pointer), size])
        self._free = _bind(driver, "cuMemFree_v2", [device_pointer])
        self._upload = _bind(driver, "cuMemcpyHtoD_v2", [device_pointer, pointer, size])
        self._download = _bind(driver, "cuMemcpyDtoH_v2", [pointer, device_pointer, size])
        self._copy = _bind(driver, "cuMemcpyDtoDAsync_v2", [device_pointer, device_pointer, size, pointer])
        self._module_load = _bind(driver, "cuModuleLoadDataEx", [address, pointer, unsigned, ctypes.POINTER(integer), address])
        self._module_unload = _bind(driver, "cuModuleUnload", [pointer])
        self._function_get = _bind(driver, "cuModuleGetFunction", [address, pointer, ctypes.c_char_p])
        self._launch = _bind(driver, "cuLaunchKernel", [pointer] + [unsigned] * 7 + [pointer, address, address])
        self._error_name = _bind(driver, "cuGetErrorName", [integer, strings])
        self._error_string = _bind(driver, "cuGetErrorString", [integer, strings])
        compiler = self._compiler
        self._nv_version = _bind(compiler, "nvrtcVersion", [ctypes.POINTER(integer), ctypes.POINTER(integer)])
        self._nv_error = _bind(compiler, "nvrtcGetErrorString", [integer], ctypes.c_char_p)
        self._program_create = _bind(compiler, "nvrtcCreateProgram", [address, ctypes.c_char_p, ctypes.c_char_p, integer, strings, strings])
        self._program_destroy = _bind(compiler, "nvrtcDestroyProgram", [address])
        self._program_compile = _bind(compiler, "nvrtcCompileProgram", [pointer, integer, strings])
        self._log_size = _bind(compiler, "nvrtcGetProgramLogSize", [pointer, ctypes.POINTER(size)])
        self._log_get = _bind(compiler, "nvrtcGetProgramLog", [pointer, pointer])
        self._binary_size = _bind(compiler, "nvrtcGetCUBINSize", [pointer, ctypes.POINTER(size)])
        self._binary_get = _bind(compiler, "nvrtcGetCUBIN", [pointer, pointer])

    def _check(self, result, operation, log=""):
        if result:
            name, description = ctypes.c_char_p(), ctypes.c_char_p()
            self._error_name(result, ctypes.byref(name))
            self._error_string(result, ctypes.byref(description))
            detail = (description.value or b"").decode("utf8", "replace")
            label = (name.value or str(result).encode()).decode("utf8", "replace")
            raise RuntimeError(f"CUDA {operation}失败：{label}（{detail}）" + (f"\n{log}" if log else ""))

    def _nvcheck(self, result, operation, log=""):
        if result:
            detail = self._nv_error(result).decode("utf8", "replace")
            raise RuntimeError(f"NVRTC {operation}失败：{detail}" + (f"\n{log}" if log else ""))

    def _current(self):
        if self._closed:
            raise RuntimeError("CUDA 运行时已关闭")
        self._check(self._set_current(self._context), "设置当前线程上下文")

    def allocate(self, nbytes: int) -> int:
        if nbytes < 0: raise ValueError("显存字节数不能为负数")
        if not nbytes: return 0
        unit = 65536 if nbytes >= 65536 else 512
        capacity = ((nbytes + unit - 1) // unit) * unit
        with self._lock:
            self._current()
            pool = self._available.get(capacity, [])
            if pool:
                pointer = pool.pop()
            else:
                value = ctypes.c_uint64()
                status = self._malloc(ctypes.byref(value), capacity)
                if status == 2:  # CUDA_ERROR_OUT_OF_MEMORY：先释放闲置缓存再重试。
                    self.empty_cache()
                    status = self._malloc(ctypes.byref(value), capacity)
                self._check(status, "分配显存")
                pointer = value.value
                self._blocks[pointer] = capacity
                self.reserved_bytes += capacity
                self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
            self._allocations[pointer] = nbytes
            self.allocated_bytes += nbytes
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.allocated_bytes)
            return pointer

    def free(self, ptr: int):
        with self._lock:
            if self._closed or ptr not in self._allocations: return
            self.allocated_bytes -= self._allocations.pop(ptr)
            # 所有后续读写仍在同一默认流，复用排在旧内核之后，不提前物理释放。
            self._available.setdefault(self._blocks[ptr], []).append(ptr)
            # 动态专家形状不能把闲置缓存堆到 WDDM 的物理显存预算之外。
            if self.reserved_bytes - self.allocated_bytes > self.cache_limit_bytes:
                self.empty_cache()

    def empty_cache(self):
        with self._lock:
            self._current()
            self._check(self._sync(), "等待闲置显存释放")
            for pointers in self._available.values():
                for ptr in pointers:
                    self._check(self._free(ptr), "释放闲置显存")
                    self.reserved_bytes -= self._blocks.pop(ptr)
            self._available.clear()

    def memory_stats(self):
        return {"allocated_bytes": self.allocated_bytes, "reserved_bytes": self.reserved_bytes,
                "peak_allocated_bytes": self.peak_allocated_bytes, "peak_reserved_bytes": self.peak_reserved_bytes}

    def reset_peak_memory_stats(self):
        self.peak_allocated_bytes = self.allocated_bytes
        self.peak_reserved_bytes = self.reserved_bytes

    def _host_buffer(self, buffer):
        view = memoryview(buffer)
        if view.readonly or not view.c_contiguous:
            raise ValueError("主机缓冲区必须连续且可写")
        return (ctypes.c_ubyte * view.nbytes).from_buffer(buffer), view.nbytes

    def upload(self, ptr: int, buffer):
        host, nbytes = self._host_buffer(buffer)
        with self._lock:
            self._current()
            if nbytes:
                self._check(self._upload(ptr, host, nbytes), "上传显存数据")


    def download(self, ptr: int, buffer):
        host, nbytes = self._host_buffer(buffer)
        with self._lock:
            self._current()
            if nbytes:
                self._check(self._sync(), "等待数据读取")
                self._check(self._download(host, ptr, nbytes), "读取显存数据")

    def copy(self, dst: int, src: int, nbytes: int):
        if nbytes < 0:
            raise ValueError("复制字节数不能为负数")
        with self._lock:
            self._current()
            if nbytes:
                self._check(self._copy(dst, src, nbytes, None), "复制显存数据")


    def synchronize(self):
        with self._lock:
            self._current()
            self._check(self._sync(), "同步设备")

    def _compile(self, source_file: Path, source: bytes, options: tuple[str, ...]):
        program = ctypes.c_void_p()
        self._nvcheck(self._program_create(ctypes.byref(program), source, source_file.name.encode("utf8"), 0, None, None), "创建编译任务")
        try:
            encoded = (ctypes.c_char_p * len(options))(*(option.encode("utf8") for option in options))
            result = self._program_compile(program, len(options), encoded)
            size = ctypes.c_size_t()
            self._nvcheck(self._log_size(program, ctypes.byref(size)), "读取编译日志大小")
            log = ctypes.create_string_buffer(size.value)
            self._nvcheck(self._log_get(program, log), "读取编译日志")
            self._nvcheck(result, f"编译 {source_file.name}", log.value.decode("utf8", "replace"))
            self._nvcheck(self._binary_size(program, ctypes.byref(size)), "读取 CUBIN 大小")
            binary = ctypes.create_string_buffer(size.value)
            self._nvcheck(self._binary_get(program, binary), "读取 CUBIN")
        finally:
            self._program_destroy(ctypes.byref(program))
        module = ctypes.c_void_p()
        info, error = ctypes.create_string_buffer(16384), ctypes.create_string_buffer(16384)
        settings = (ctypes.c_int * 4)(3, 4, 5, 6)
        values = (ctypes.c_void_p * 4)(ctypes.addressof(info), len(info), ctypes.addressof(error), len(error))
        result = self._module_load(ctypes.byref(module), binary, 4, settings, values)
        log = (info.value + error.value).decode("utf8", "replace")
        self._check(result, f"加载内核 {source_file.name}", log)
        return module

    def launch(self, source_file: Path, name: str, size: int, arguments: list,
               options: tuple[str, ...] = (), block_size: int = 256, *, grid=None, block=None, shared_memory=0):
        if size < 0 or not 1 <= block_size <= 1024:
            raise ValueError("内核元素数不能为负数，线程块大小必须在 1 到 1024 之间")
        if size == 0: return
        source_file = Path(source_file)
        function_key = (source_file, tuple(options), name)
        with self._lock:
            self._current()
            if function_key not in self._functions:
                if source_file not in self._sources:
                    source = source_file.read_bytes()
                    self._sources[source_file] = (source, hashlib.sha256(source).digest())
                source, digest = self._sources[source_file]
                architecture = "".join(map(str, self.compute_capability))
                settings = ("--std=c++17", f"--gpu-architecture=sm_{architecture}", f"--include-path={source_file.parent}")
                settings += tuple(f"--include-path={directory}" for directory in self._includes) + tuple(options)
                key = (digest, settings)
                if key not in self._modules: self._modules[key] = self._compile(source_file, source, settings)
                function = ctypes.c_void_p()
                self._check(self._function_get(ctypes.byref(function), self._modules[key], name.encode("utf8")), f"查找内核 {name}")
                self._functions[function_key] = function
            parameters = (ctypes.c_void_p * len(arguments))(*(ctypes.addressof(argument) for argument in arguments))
            grid = grid or ((size + block_size - 1) // block_size, 1, 1)
            block = block or (block_size, 1, 1)
            if len(grid) != 3 or len(block) != 3 or any(n <= 0 for n in (*grid,*block)) or block[0]*block[1]*block[2] > 1024:
                raise ValueError("CUDA 网格和线程块必须是合法的三维正整数")
            self._check(self._launch(self._functions[function_key], *grid, *block, shared_memory, None, parameters, None), f"启动内核 {name}")
            self.launches += 1
            if self.debug_sync: self._check(self._sync(), f"等待内核 {name}")

    def close(self):
        """退出时尽力回收所有资源；已释放的存储随后调用 free 也安全。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._context.value:
                self._set_current(self._context)
                self._sync()
                for pointer in self._blocks:
                    self._free(pointer)
                for module in self._modules.values():
                    self._module_unload(module)
                self._release(self._device)
                self._context = ctypes.c_void_p()
            self._allocations.clear()
            self._blocks.clear()
            self._available.clear()
            self._sources.clear()
            self.allocated_bytes = self.reserved_bytes = 0
            self._modules.clear()
            self._functions.clear()
            for directory in self._dll_directories:
                directory.close()
            self._dll_directories.clear()


def get_runtime() -> Runtime:
    """首次使用 CUDA 时加载 DLL 并初始化设备；导入 qutai 不触发这里。"""
    global _runtime
    with _runtime_lock:
        if _runtime is None or _runtime._closed:
            _runtime = Runtime()
        return _runtime
