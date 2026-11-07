# Qutai 示例

示例按四条学习路线组织，全部只使用 Python 标准库与 Qutai。先在项目根目录用 Python 3.14 环境执行 `python -m pip install -e .`，再在 PyCharm 或 Jupyter 中从上到下运行 Notebook。

| 目录 | 内容 | 建议顺序 |
| --- | --- | --- |
| [nn](nn/README.md) | 基础神经网络与图像分类 | 回归 → Iris → LeNet → ResNet；再看 Attention、ViT |
| [generation](generation/README.md) | 扩散模型生成 | DiT 加噪训练、逐步去噪；Flow Matching 对照 |
| [lm](lm/README.md) | 语言模型与训练入口 | 字符 LM → Dense/MoE Decoder → 真实语料 CLI |
| [quantum](quantum/README.md) | 量子态、量子优化与 PQC | Bell → 可训练电路 → VQE/QAOA → 混合网络与 barren plateau |

默认 `device="cuda"`、`dtype="float32"`，无 CUDA 时把设备改成 `"cpu"`。新的 Decoder 示例在 CUDA 上使用 BF16 激活，参数、参数梯度与 Adam 状态仍为 FP32；CPU 使用普通 FP32。量子态对应 complex64。SDK 不依赖 PyTorch、NumPy、Triton 或其他量子 package；CUDA 需要 NVIDIA 驱动和含 NVRTC 的 Toolkit。

Notebook 可以从项目根目录、`examples` 或各分组目录启动。图像数据解析集中在 [datasets.py](datasets.py)，Iris、文本与下载缓存共用 [data](data/README.md)，不在各组重复复制。Iris 与字符语料已随源码提供；图像缓存不存在时才下载。

共有16本Notebook。教学示例默认规模如下，少量更新用于理解接口，不代表正式训练质量：

| Notebook | 默认运行 |
| --- | --- |
| [nn/regression](nn/regression.ipynb) | 线性回归301步、拟合平方1001步、保存/恢复预测 |
| [nn/iris](nn/iris.ipynb) | Iris三分类501步，独立测试与标准化 |
| [nn/lenet](nn/lenet.ipynb) | MNIST训练2000张×5轮，测试500张 |
| [nn/resnet](nn/resnet.ipynb) | CIFAR-10训练500张×3轮，测试100张 |
| [nn/attention](nn/attention.ipynb) | 401步学习读取序列中被标记的数值 |
| [nn/vit](nn/vit.ipynb) | Patch/Transformer图像分类2步 |
| [generation/diffusion](generation/diffusion.ipynb) | DiT扩散训练2轮×2批，采样8步 |
| [lm/language_model](lm/language_model.ipynb) | 字符LM训练2步、续写、KV cache对照 |
| [lm/decoder_training](lm/decoder_training.ipynb) | Dense/MoE各20步，核对同一个下一步恢复 |
| [quantum/bell](quantum/bell.ipynb) | Bell态、概率、采样、坍缩与Bloch球 |
| [quantum/quantum_training](quantum/quantum_training.ipynb) | 一个RY角度训练2步 |
| [quantum/hybrid](quantum/hybrid.ipynb) | Linear→PQC→Linear混合网络2步 |
| [quantum/vqe](quantum/vqe.ipynb) | 两比特VQE，能量优化2步 |
| [quantum/qaoa](quantum/qaoa.ipynb) | 四节点MaxCut，QAOA更新2步 |
| [quantum/quantum_probabilities](quantum/quantum_probabilities.ipynb) | 可求导概率分布训练2步 |
| [quantum/barren_plateau](quantum/barren_plateau.ipynb) | 多比特/深度/seed的真实梯度统计与解析对照 |

回归模型写到当前工作目录的 `runs/regression.json`。Decoder恢复演示使用自动清理的临时目录；正式CLI训练必须显式选择新的输出目录。源Notebook保留已有教学输出；重新执行时显示当前模型和设备的实际结果。

LM主入口是 `python examples/lm/train.py --help`；旧 `examples/lm_train.py`保留为简短兼容入口。量子梯度性能对照位于 `python examples/quantum/adjoint_benchmark.py`。
