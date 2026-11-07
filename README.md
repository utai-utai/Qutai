<p align="center"><img src="logo.png" alt="Qutai" width="240"></p>

# Qutai

从零实现的数值计算、神经网络与量子计算 SDK。支持 CPU 和 NVIDIA CUDA，只依赖 Python 标准库，CUDA 使用自写内核。

## 安装

需要 Python 3.14+。在项目根目录执行：

```bash
python -m pip install -e .
```

CPU 无需额外配置；CUDA 需要 NVIDIA 驱动和含 NVRTC 的 CUDA Toolkit。

## 快速使用

```python
from qutai import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(loss.item(), x.grad.tolist())  # 14.0 [2.0, 4.0, 6.0]
```

用 `device="cuda", dtype="float32"` 创建 Tensor，或用 `tensor.to(...)` / `model.to(...)` 迁移设备和精度。训练循环可用 `backward(retain_graph=False)` 及时释放计算图。

## 模块

源码位于 `src/qutai`，`backends` 提供 CPU/CUDA 实现。

| 模块 | 内容 |
| --- | --- |
| `core` | Tensor、自动求导、数学运算、Stream 批处理 |
| `nn` | 网络层、损失、优化器、保存与加载 |
| `lm` | 字符 LM、Dense/MoE Decoder、BPE、训练与断点恢复 |
| `generation` | Diffusion、Flow Matching、DiT |
| `quantum` / `qnn` | 量子态、电路、可求导读出、QAOA、PQC |
| `visualization` | 曲线、表格、图像、网络、电路与 Bloch 球 |

`lm.Decoder` 可用 `lm.autocast()` 开启 CUDA BF16 激活，参数、参数梯度和 Adam 状态保持 FP32；其他示例使用 float32，量子态使用 complex64。

## 示例

在安装好 Qutai 的环境中运行 Notebook；默认 CUDA，可在第一格改为 CPU。

| 目录 | 示例 |
| --- | --- |
| [nn](examples/nn/) | [回归](examples/nn/regression.ipynb)、[Iris](examples/nn/iris.ipynb)、[LeNet](examples/nn/lenet.ipynb)、[ResNet](examples/nn/resnet.ipynb)、[Attention](examples/nn/attention.ipynb)、[ViT](examples/nn/vit.ipynb) |
| [generation](examples/generation/) | [扩散生成](examples/generation/diffusion.ipynb) |
| [lm](examples/lm/) | [字符 LM](examples/lm/language_model.ipynb)、[Dense/MoE 训练与恢复](examples/lm/decoder_training.ipynb)、[训练脚本](examples/lm/train.py)、[配置](examples/lm/configs/) |
| [quantum](examples/quantum/) | [Bell](examples/quantum/bell.ipynb)、[VQE](examples/quantum/vqe.ipynb)、[QAOA](examples/quantum/qaoa.ipynb)、[PQC 训练](examples/quantum/quantum_training.ipynb)、[概率训练](examples/quantum/quantum_probabilities.ipynb)、[混合网络](examples/quantum/hybrid.ipynb)、[barren plateau](examples/quantum/barren_plateau.ipynb) |

[examples/data.py](examples/data.py) 用标准库读取数据，用 Qutai Tensor/Stream 组织样本；不依赖 datasets、NumPy 或 PyTorch。缓存统一放在 `examples/data`，MNIST/CIFAR-10 缺失时自动下载。示例默认是教学规模。

数据来源：[Iris / Fisher 1936](https://doi.org/10.24432/C56C76)（[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，原数据增加表头并简化品种名）、[MNIST](https://github.com/cvdfoundation/mnist)、[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html)。

## 开发

```bash
python -X utf8 -m unittest discover -s tests
python examples/lm/train.py --help
```
