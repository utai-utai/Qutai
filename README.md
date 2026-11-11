<p align="center"><img src="logo.png" alt="Qutai" width="240"></p>

# Qutai

从零实现的轻量计算 SDK，涵盖数值计算、神经网络与量子计算。无第三方 Python 运行依赖，支持 CPU、Apple GPU 和 NVIDIA CUDA。

[使用手册](https://qutai.yaoyuhan.com) · [示例教程](https://github.com/utai-utai/Qutai/blob/main/examples/README.md)

## 安装

需要 **Python 3.14+**。

```bash
pip install qutai
```

## 快速开始

用普通列表创建 Tensor，完成计算后调用 `backward()` 求导：

```python
from qutai import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.item())     # 14.0
print(x.grad.tolist())  # [2.0, 4.0, 6.0]
```

## 模块

| 入口 | 功能 |
| --- | --- |
| `Tensor`、`Stream` 与数学函数 | 张量运算、自动求导与批次数据流 |
| `nn` | 网络层、损失函数、优化器与模型保存加载 |
| `lm` | 字符语言模型、Dense / MoE Decoder、分词与训练 |
| `generation` | Diffusion、Flow Matching 与 DiT |
| `quantum` | 量子态、量子门、电路、测量与 QAOA |
| `qnn` | 参数化量子电路 PQC，与经典网络组合训练 |
| `visualization` | 曲线、表格、图片、网络结构、量子电路与 Bloch 球 |

## 设备

默认使用 CPU；在创建 Tensor、网络层或模型时传入 `device=` 即可选择设备，例如 `Tensor([1, 2, 3], device="apple")`、`nn.Model([...], device="cuda")`。模型会统一内部各层的设备。

| `device` | 运行要求 | 默认精度（实数 / 复数） |
| --- | --- | --- |
| `"cpu"` | 无需额外配置 | float64 / complex128 |
| `"apple"` | Apple Silicon、macOS 13+、Xcode Command Line Tools | float32 / complex64 |
| `"cuda"` | Windows 或 Linux、NVIDIA 驱动、含 NVRTC 的 CUDA Toolkit | float32 / complex64 |

Apple 后端首次使用时自动编译桥接库。通常无需设置 `dtype`，有特殊需求时可显式指定。

## 源码开发

在项目根目录安装并运行测试：

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

教程从回归、分类逐步介绍卷积网络、Transformer、语言模型、图像生成与量子计算。