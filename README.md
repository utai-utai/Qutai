<p align="center"><img src="logo.png" alt="Qutai" width="240"></p>

# Qutai

从零实现的轻量计算 SDK：数值计算、神经网络与量子计算。只依赖 Python 标准库，支持 CPU、Apple GPU 和 NVIDIA CUDA。

## 安装

需要 Python 3.14+。从源码开发，在项目根目录运行：

```bash
python -m pip install -e .
```

CPU 无需额外配置。Apple GPU 需要 Apple Silicon、macOS 13+ 和 Xcode Command Line Tools；首次使用自动编译桥接库。CUDA 需要 NVIDIA 驱动及含 NVRTC 的 CUDA Toolkit。

## 开始使用

```python
from qutai import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.tolist())  # [2.0, 4.0, 6.0]
```

`Tensor(..., device="apple")` 选择设备，`model.to(device)` 迁移模型。设备可选 `cpu / apple / cuda`，精度自动选择：CPU 默认 float64 / complex128，Apple 和 CUDA 默认 float32 / complex64。跨设备迁移时采用目标设备的默认精度，同设备保持现有精度；有特殊需求时仍可显式指定 `dtype`。

## 模块

| 模块 | 功能 |
| --- | --- |
| `Tensor`、根入口数学函数 | 数值运算与自动求导 |
| `nn` | 神经网络层、损失、优化器与保存加载 |
| `lm` | 字符语言模型、Dense/MoE Decoder 与训练工具 |
| `generation` | Diffusion、Flow Matching 与 DiT |
| `quantum / qnn` | 量子态、电路、QAOA 与参数化量子层 |
| `visualization` | 曲线、表格、图片、网络、电路与 Bloch 球 |

## 教程

从 [examples/README.md](examples/README.md) 开始。每本只讲一件事，按数据、模型、训练、结果的顺序展开。DiT 教程与语言模型一起放在 `examples/lm`，SDK 的生成接口仍在 `qutai.generation`。

```bash
python -m unittest discover -s tests
```
