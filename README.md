<p align="center">
  <img src="logo.png" alt="Qutai" width="380">
</p>

<big align="center"><strong>从零手写的轻量计算 SDK：数值计算、神经网络与量子计算。</strong></b>
---

Python 标准库实现，无第三方运行依赖。支持 CPU 和 NVIDIA CUDA 后端；CUDA 的实数和复数内核在 GPU 上执行。

## 安装

需要 **Python 3.14 或以上**。

```bash
pip install qutai
```

## 快速开始

直接用列表创建 Tensor，运算时自动记录梯度：

```python
from qutai import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss = (x + x).sum()
loss.backward()

print(loss.item())     # 12.0
print(x.grad.tolist()) # [2.0, 2.0, 2.0]
```

构造并运行一个 Bell 态电路：

```python
from qutai import quantum

circuit = quantum.Circuit(2, [quantum.H(0), quantum.CNOT(0, 1)])
state = circuit.run()
print(state.probabilities())  # 约 [0.5, 0.0, 0.0, 0.5]
```

## 模块

| 模块 | 功能 |
| --- | --- |
| `Tensor`、根入口数学函数 | 广播、矩阵运算、卷积与自动求导 |
| `nn` | 神经网络层、Transformer、损失、优化器与保存加载 |
| `lm` | 字符模型与 KV cache；Dense/MoE 解码器、BPE、分片数据与可恢复训练 |
| `generation` | Diffusion、Flow Matching 与 DiT |
| `quantum` | 量子态、门、电路、测量、可求导读出与 QAOA |
| `qnn` | 参数化量子电路 PQC，与经典网络混合训练 |
| `visualization` | 曲线、表格、图片、网络、电路与 Bloch 球 |

## 示例与开发

[Notebook 示例](https://github.com/utai-utai/Qutai/blob/main/examples/README.md)涵盖回归、Iris、LeNet、ResNet、ViT、语言模型、DiT、Bell 态、混合网络、VQE 和 QAOA。量子训练支持伴随求导与参数移位对照；多数较大模型仅训练两步验证流程。

从源码安装并运行测试：

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

## CUDA（Windows / NVIDIA）

Python SDK 仍然没有第三方运行依赖。CUDA 通过标准库 `ctypes` 调用 NVIDIA Driver API 和 NVRTC；需要另外安装 NVIDIA 驱动和 CUDA Toolkit（含 NVRTC）。普通 CPU 导入不加载 CUDA DLL，也不需要安装 Toolkit。

```python
from qutai import Tensor, nn, is_available

assert is_available("cuda")
model = nn.Model([nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)])
model.to("cuda", dtype="float32")
optimizer = nn.AdamW(model, learning_rate=0.01)
for _ in range(2):
    optimizer.zero_grad()
    loss = nn.MSE()(model([[1, 2], [2, 1]]), [[3], [3]])
    loss.backward()
    optimizer.step()
    print(loss.item())
```

沿用统一 `Tensor(..., device=..., dtype=...)`、`tensor.to(...)` 和 `model.to(...)` 接口。已有 Tensor 输入必须与模型同设备、同精度，例如 `inputs.to("cuda", dtype="float32")`；普通数字或列表沿用模型参数配置。先迁移模型，再建立新的前向图；已有优化器状态会在 `step()` 跟随参数迁移。`tolist()` / `item()`、可视化、模型保存、采样输出是明确的数据读取出口。

CUDA 提供 float32 / float64、complex64 / complex128、bfloat16 和内部 int64 存储；建议先用 float32，量子态对应 complex64。基础 Tensor 运算与 `to(..., dtype="bfloat16")` 支持真实 2 字节 BF16 存储。CNN、量子模块及旧 `nn` 模块的 BF16 路径尚未全覆盖，继续使用原有 float32 / float64 或复数精度。CUDA 覆盖基础运算、自动求导、CNN、归一化、Transformer / KV cache、生成模型、量子态与可训练量子电路。计算使用自有 `.cu` 内核，无 CPU 计算回退；当前为单 GPU 默认流异步执行，包含自有分块 GEMM 与 BF16 Tensor Core 优化。

首次启动某类内核时，NVRTC 按当前设备的 SM 编译原生 CUBIN，进程内缓存模块；这避免新 Toolkit 的 PTX 版本超过驱动 JIT 支持版本。`CUDA_PATH` / `CUDA_HOME` 可指定 Toolkit；Windows 也检查标准安装目录。发布包携带 `.cu` 源码。

本机验证环境：Python 3.14.0（64 位）、Windows、RTX PRO 4000 Blackwell（SM120，约 24 GB）、驱动 596.72、CUDA / NVRTC 13.4。其他 GPU 尚未实卡验证。

Windows 开发与测试使用 UTF-8 模式（PyCharm 也可设置 `PYTHONUTF8=1`），以免默认 GBK 无法显示进度条或读取 SVG：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

GPU 测试在没有 CUDA 的环境中跳过；CPU CI 的通过不代表实卡验证。NVIDIA 接口依据：[NVRTC](https://docs.nvidia.com/cuda/nvrtc/)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)。

## Dense / MoE 与量子训练底座

本轮能力目前随开发源码和本地构建包提供；使用开发源码请先执行 `pip install -e .`。

语言模型结构、语料、训练循环与二进制 checkpoint 集中在 `src/qutai/lm/`，用法见 [LM 说明](src/qutai/lm/README.md)。原有字符 `lm.Model` 与 Notebook 继续使用；`lm.Config` / `lm.Decoder` 用于更大的 Dense / MoE。SDK 的 Python 运行依赖仍为空。

```python
from qutai import lm, nn

config = lm.Config(vocab_size=256, n_layers=2, d_model=64, n_heads=4,
                   n_kv_heads=2, d_ff=192, max_seq_len=64)
model = lm.Decoder(config, device="cuda", seed=42)
optimizer = nn.AdamW(model, learning_rate=0.001)
for _ in range(2):
    optimizer.zero_grad()
    with lm.autocast():
        loss = model.loss([[1, 2, 3, 4], [4, 3, 2, 1]])
    loss.backward(retain_graph=False)
    optimizer.step()
    print(loss.item())
```

CUDA `lm.Decoder` 在 `lm.autocast()` 下将 QKV、attention、FFN 和输出 head 的激活存为真实 BF16，并使用自写 BF16 Tensor Core GEMM、FP32 累加。模型参数、参数梯度与 Adam 状态保留 FP32；residual hidden、Norm、Router 线性投影及路由计算、Loss 使用 FP32。已有 BF16 GEMM 输入在退出 autocast 后仍使用 Tensor Core，反向不会依赖调用时的上下文。现有默认 float32 / float64 路径仍保留。

CUDA 内核在同一默认 stream 异步排队，显存缓存按同流顺序复用；`tolist()` / `item()` / `synchronize()` 和 checkpoint 读回明确等待完成。`QUTAI_CUDA_SYNC=1` 可用于逐内核调试。`cuda.memory_stats()` 区分当前活动分配和保留缓存，`cuda.empty_cache()` 只释放闲置缓存。闲置缓存预算为 1 GiB，动态形状超过预算后会同步回收，避免 Windows 下保留分配挤占显存。

`backward()` 默认保留计算图以兼容重复反向；长期训练调用 `backward(retain_graph=False)` 及时释放，已经释放的图不能再次反向。`lm.activation_checkpoint` 重计算确定性、无副作用函数，模型闭包参数需通过 `parameters=` 显式提供。

量子 `qnn.PQC` 的 CUDA 路径按批次处理 `(..., qubits)`，用共享执行计划、GPU 伴随反向和工作态分块；每个样本仍是独立量子态。该接口不预设跨 token 的量子记忆结构。状态空间仍按 `2**qubits` 增长，批处理不改变这一规模关系。

示例命令行 `examples/lm_train.py` 与三个结构配置提供新训练、完整更新边界保存和显式恢复。安装包携带所有自写 CUDA 源码；PyTorch / tokenizers 只用于独立测试对照，不是 SDK 运行依赖。
