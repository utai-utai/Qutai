<p align="center"><img src="logo.png" alt="Qutai" width="240"></p>

# Qutai

从零实现的轻量计算 SDK：数值计算、神经网络与量子计算。只依赖 Python 标准库，支持 CPU、Apple GPU 和 NVIDIA CUDA。

[使用手册](https://qutai.yaoyuhan.com)

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

统一在创建时指定设备，例如 `Tensor(..., device="apple")`、`nn.Linear(2, 1, device="apple")` 或 `nn.Model([...], device="apple")`。设备可选 `cpu / apple / cuda`，精度自动选择：CPU 默认 float64 / complex128，Apple 和 CUDA 默认 float32 / complex64。模型会统一内部各层的设备；有特殊需求时仍可显式指定 `dtype`。

展平、池化、激活和求平均等基础运算层可直接接收数据，例如 `nn.Flatten(x)`、`nn.GlobalAvgPool2d(x)`、`nn.ReLU(x)`。组装模型时省略数据，仍写 `nn.Model([nn.ReLU(), nn.Flatten()])`。

## 模块

| 模块 | 功能 |
| --- | --- |
| `Tensor`、根入口数学函数 | 数值运算与自动求导 |
| `nn` | 神经网络层、损失、优化器与保存加载 |
| `lm` | 字符语言模型、Dense/MoE Decoder 与训练工具 |
| `generation` | Diffusion、Flow Matching 与 DiT |
| `quantum / qnn` | 量子态、电路、QAOA 与参数化量子层 |
| `visualization` | 曲线、表格、图片、网络、电路与 Bloch 球 |

## 公共张量与权重接口

普通切片 x[..., ::2] 保留自动求导；CUDA 根据形状和步长在设备上索引。nn.Embedding 直接按行查表，避免生成逐元素主机索引。

```python
from qutai import Tensor, SafeTensors, concatenate, save_safetensors

x = Tensor([[1., 2., 3., 4.]], requires_grad=True)
a, b = x.split(2, axis=-1)  # 等分，轴长度必须整除份数
concatenate((a, b), axis=-1).sum().backward()
save_safetensors("weights.safetensors", {"weight": x})
weights = SafeTensors("weights.safetensors", device="cpu")
x.copy_(weights["weight"])
```

SafeTensors 按需加载并缓存 F32/F64/BF16 权重，metadata 为字符串字典；BF16 需要 CUDA。tanh、atan、nn.Dropout、set_recording / set_training 和 lm.linear / rope / silu 也可直接使用。GPU 索引、查表和权重上传不需要第三方 Python 包。

本次实现位置：core/shape.py 与 core/operations/transforms.py 处理索引和拼接；core/tensor.py 提供 split / copy_；backends/cuda.py 与 backends/cuda/tensor.cu 执行 CUDA 索引；core/serialization.py 负责权重读写。CPU/CUDA 前向、梯度与文件往返覆盖在 tests/test_public_tensor_io.py。

CUDA 通用算子会缓存最多 128 项只读索引参数；较宽的 LayerNorm / 连续轴 Softmax 使用并行归约，较大的 FP32 矩阵使用 64×64 分块。Python 常数除法在主机检查除零，避免 GPU 状态读回；Tensor 除数保持原有检查。相关实现位于 backends/cuda.py、backends/cuda/nn.cu、backends/cuda/gemm.cu 和 core/operations/elementwise.py，覆盖在 tests/test_cuda_parallel_rows.py。

## 教程

从 [examples/README.md](examples/README.md) 开始。每本只讲一件事，按数据、模型、训练、结果的顺序展开。DiT 教程与语言模型一起放在 `examples/lm`，SDK 的生成接口仍在 `qutai.generation`。

```bash
python -m unittest discover -s tests
```
