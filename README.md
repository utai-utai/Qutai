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
| `lm` | 字符语言模型、随机续写与 KV cache |
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

CUDA 提供 float32 / float64、complex64 / complex128 和内部 int64 存储；建议先用 float32，量子态对应 complex64。覆盖基础运算、自动求导、CNN、归一化、Transformer / KV cache、生成模型、量子态与可训练量子电路。计算使用自有 `.cu` 内核，无 CPU 计算回退；当前为单 GPU 同步执行，尚未做性能优化。

首次启动某类内核时，NVRTC 按当前设备的 SM 编译原生 CUBIN，进程内缓存模块；这避免新 Toolkit 的 PTX 版本超过驱动 JIT 支持版本。`CUDA_PATH` / `CUDA_HOME` 可指定 Toolkit；Windows 也检查标准安装目录。发布包携带 `.cu` 源码。

本机验证环境：Python 3.14.0（64 位）、Windows、RTX PRO 4000 Blackwell（SM120，约 24 GB）、驱动 596.72、CUDA / NVRTC 13.4。其他 GPU 尚未实卡验证。

Windows 开发与测试使用 UTF-8 模式（PyCharm 也可设置 `PYTHONUTF8=1`），以免默认 GBK 无法显示进度条或读取 SVG：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

GPU 测试在没有 CUDA 的环境中跳过；CPU CI 的通过不代表实卡验证。NVIDIA 接口依据：[NVRTC](https://docs.nvidia.com/cuda/nvrtc/)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)。
