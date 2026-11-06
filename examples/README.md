# qutai 教学示例

在 PyCharm 选择项目的 Python 3.14 环境，以 `examples` 为工作目录，从上到下执行单元格。这样可以直接读取 `data/`，图片示例也能直接导入同目录的 `datasets.py`。

经典模型示例按「看数据 → 搭模型 → 训练 → 预测／可视化」展开。训练循环显式保留清梯度、损失、反向传播和更新参数四步；可视化统一通过 `vis` 调用。代码只使用标准库和 qutai。

| Notebook | 学习内容 | 默认训练 |
| --- | --- | --- |
| `regression.ipynb` | 线性关系、拟合 x²、画曲线与保存加载 | 301 / 1001 步 |
| `classification.ipynb` | Iris 三分类、划分测试集、标准化、类别与概率 | 500 步 |
| `lenet.ipynb` | MNIST、卷积、池化与展平 | 2000 张，5 轮 |
| `resnet.ipynb` | CIFAR-10、残差块与 BatchNorm | 500 张，3 轮 |
| `attention.ipynb` | 通过注意力读取被标记的数值 | 400 步 |
| `vit.ipynb` | 图片分块、Transformer 与分类 | 2 步 |
| `language_model.ipynb` | 字符编号、因果注意力、AdamW、梯度裁剪、随机续写与 KV cache | 2 步 |
| `dit.ipynb` | Stream 准备生成目标，DiT 预测速度与 Euler 采样 | 2 批，采样 8 步 |
| `bell.ipynb` | 量子态、H/CNOT、电路与概率图、采样及测量坍缩 | 不涉及训练 |
| `quantum_training.ipynb` | Tensor 旋转角度、期望值、量子反向与 SGD | 2 步 |
| `hybrid.ipynb` | Linear → 随机结构 PQC → Linear，端到端混合网络 | 2 步 |
| `vqe.ipynb` | 组合哈密顿量、多结果读出、能量优化与局部 Bloch 球 | 2 步 |
| `qaoa.ipynb` | 四节点 MaxCut、成本与混合演化、分组概率和切分图 | 2 步 |
| `quantum_probabilities.ipynb` | 用可求导概率学习目标分布、边缘概率与反向传播 | 2 步 |

两步实验用于理解和验证流程，预测或生成效果需要后续训练。LeNet / ResNet 保留原训练规模；本次精简后只以少量批次验证，所以完整训练与测试单元的输出留空，运行后会显示你的结果。

图像和文本统一用 `from qutai import Stream` 分批。`data = Stream(...)` 创建一次，`for inputs, targets in data` 每次遍历一轮；`len(data)` 是批次数。图像形状来自当前批第一张图片，调用时不用传 `shape`；同批图片尺寸不一致时在外部预处理。训练流默认打乱并显示进度，测试或预览设置 `shuffle=False, progress=False`。

DiT 使用 `Stream(images, method=method)`，原生支持 Diffusion 和 FlowMatching，自动完成 0–1 到 −1–1 的范围转换，直接输出 `(inputs, times, targets)`。不需要回调函数；`seed` 可统一复现打乱、时间与噪声。多张图片统一用 `visualization.ImageCombine(...)` 组合显示。

具体数据集的解析和预处理继续留在 `datasets.py`，不纳入 SDK。其 `Images` 集合只保存原始像素字节，`images[i]` 取样时返回归一化 Tensor，可以直接查看 `.shape`。SDK 的 `Stream` 同样可以接收普通列表或其他可索引的样本集合。

仅回归示例保存最后训练的平方模型到 `regression.json`，其他示例结束于预测或可视化。重复的梯度和数值检查交给 `tests`，教学单元集中展示算法流程。

`bell.ipynb` 从全零态构造 Bell 态，图表和采样结果均已保留，可直接阅读。q0 对应位串最左侧的一位；`sample` / `counts` 保留原状态，`measure` 会使状态坍缩。最后用 Bloch 球对照 H 门后的局部纯态与 Bell 态中位于球心的局部混合态。示例只使用标准库和 qutai，无需下载数据。

`quantum_training.ipynb` 将单个 RY 门的角度交给 SGD，使用 MSE 学习目标 Z 期望值 0.5。`circuit.run()` 查看数值量子态，`circuit.expectation(...)` 保留可训练角度的计算图；两次更新的记录和图表均已保留。

`hybrid.ipynb` 将 `qnn.PQC` 直接放入 `nn.Model`。每个旋转位置的 RX/RY/RZ 在创建时随机选择并固定，训练更新角度。示例使用九对普通数字，已保留混合网络图、实际电路、两次更新的损失及预测，不需要数据下载。

`vqe.ipynb` 用一个 RY 角度与 CNOT 准备两比特状态，直接最小化组合哈密顿量的期望值。理论最低能量为 `-sqrt(2)`，两步 SGD 后的能量、概率图与 q0 的 Bloch 球均已保留；只验证流程，不要求两步收敛。

`qaoa.ipynb` 将四节点环的 MaxCut 编码为成本，使用一层 QAOA 的 gamma、beta 两个参数。严格两次更新后期望切边数为 `1.421633 → 2.585731 → 2.935219`；理论最大值是 4，最优位串为 `0101` 和 `1010`。Notebook 逐步展示无向图、16 种分组的得分、电路、训练、概率、采样和按组着色的图；图表均已保留，不需要数据下载或第三方包。

量子训练默认使用伴随求导；`circuit.expectation(..., method="parameter-shift")` 可显式选择参数移位核对梯度。外部仍然调用 `loss.backward()`。性能对照单独放在 `adjoint_benchmark.py`，可在项目根目录运行 `python examples/adjoint_benchmark.py`：双方各预热一次，再交替顺序各测三次，比较前向加反向的中位耗时及最大梯度差，不执行参数更新。

`quantum_probabilities.ipynb` 直接将 `circuit.probabilities()` 返回的 Tensor 交给 MSE，用一个角度学习 `[0.15, 0, 0, 0.85]`。两步更新的 MSE 为 `0.192294 → 0.126238 → 0.074218`；电路、概率和损失曲线均已保留。部分比特可以用 wires 指定，概率同样支持伴随法和参数移位反向；数值 `state.probabilities()` 保留普通列表接口。

`language_model.ipynb` 用 AdamW 更新两步，在 backward 与 step 之间执行全局梯度裁剪，并对照默认贪心续写与 temperature/top_k 随机采样。示例 seed=42 用于复现，SDK 默认 seed=None；这里不保存模型或优化器文件，恢复训练的用法见项目 README。

语言模型的 generate 默认启用 KV cache，Notebook 最后用 use_cache=False 对照计算结果和窗口未满时的投影处理量。固定绝对位置的窗口移满后会重建缓存，保证与原滑窗语义一致；缓存不保存在模型中。
