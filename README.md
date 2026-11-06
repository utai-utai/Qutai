# qutai

从零手写的轻量计算 SDK，逐步实现数值计算、机器学习、语言模型与量子计算。

当前开发环境：Python 3.14。

Notebook 的运行方式与默认训练量见 [示例说明](https://github.com/utai-utai/Qutai/blob/main/examples/README.md)。以 `examples` 为工作目录，从上到下运行；各本统一按数据、模型、训练、预测和可视化展开。

## 当前模块

数学函数直接从 `qutai` 根入口导出；`core.tensor` 处理输入与结果包装，`core.operations` 按功能组织前向计算和局部反向规则，`backends.cpu` 执行数值计算。`Tensor` 保存形状与一维浮点数数据。例如：

```python
from qutai import add, argmax

print(add([1, 2], [3, 4]))  # [4.0, 6.0]
print(argmax([0.1, 0.8, 0.2]))  # 1
```

也可以 `import qutai as qt`，通过 `qt.add(...)`、`qt.sum(...)` 调用，避免与 Python 内置的 `sum` 重名。

`qutai.nn` 放置神经网络组件，当前提供 `Layer`、`Linear`、`Conv2d`、`MaxPool2d`、`Flatten`、`BatchNorm2d`、`Identity`、`Residual`、`ResNetBlock`、`GlobalAvgPool2d`、`ReLU`、`Softmax`、`Attention`、Transformer 相关组件、顺序模型 `Model`、均方误差 `MSE`、交叉熵 `CrossEntropy` 和优化器 `SGD`、`AdamW`。字符语言模型相关接口位于独立的 `qutai.lm` 模块，整数查表层 `Embedding` 放在通用 `nn` 中。

`qutai.quantum` 提供复数纯量子态、基础门、顺序电路、组合观测量、可求导期望与概率、QAOA、采样与测量坍缩；电路、概率和 Bloch 球统一放在 `qutai.visualization`。

`qutai.qnn` 提供量子神经网络组件；当前的 `PQC` 将固定随机结构的参数化量子电路封装成 Tensor 到 Tensor 的层，可直接插入 `nn.Model`。

`nn/layers.py` 定义薄基类 `Layer`、具体层，以及 `Residual`、`ResNetBlock` 组合层；`nn/model.py` 负责按顺序组合各层，提供 `model(inputs, train=...)`、`model.save(path)`、`model.load(path)` 入口。`nn/serialization.py` 负责结构描述、文件读写和模型重建。

内部的 `Sequence` 共用顺序执行与参数收集逻辑，供 Model 和 ResNetBlock 使用；对外继续通过 `nn.Model` 搭建模型。

`core/context.py` 统一管理两个独立状态：`training` 决定 BatchNorm 等层的训练或预测行为，`recording` 决定是否记录计算图。模型调用时通常一起设置，但含义不同；`core/grad.py` 只负责通过 `record` 登记计算关系和通过 `backward` 传播梯度。

| 运算文件 | 功能 |
| --- | --- |
| `core/operations/elementwise.py` | 广播四则运算、取负、平方、ReLU、GELU、指数、自然对数 |
| `core/operations/linalg.py` | 线性代数（linear algebra），向量、矩阵和广播批次矩阵乘法 |
| `core/operations/reductions.py` | sum、mean、argmax，支持 axis 和 keepdims |
| `core/operations/normalization.py` | Softmax、BatchNorm2d、LayerNorm 及各自的反向规则 |
| `core/operations/transforms.py` | 任意维轴置换、reshape、索引与切片 |
| `core/operations/convolution.py` | NCHW 二维卷积、最大池化及各自的反向规则 |

使用项目虚拟环境运行测试：

```bash
python -m unittest discover -s tests -v
```

## 存储与 CPU 后端

列表输入、模型调用和 `loss.backward()` 的用法保持不变。Tensor 内部使用 float64 存储，量子 State 使用 complex128 振幅存储；均通过标准库 `array` 保存为连续内存，复数按实部、虚部交错排列。池化反向需要的位置缓存使用内部 int64 存储，实数 Tensor 暂只提供 float64。缓冲区接口见 [Python 文档](https://docs.python.org/3.14/library/array.html)。

| 文件 | 职责 |
| --- | --- |
| `backends/storage.py` | 存储约定：设备、精度、元素数、复制与普通数值导出，不依赖 Tensor |
| `backends/device.py` | 根据设备或输入存储选择后端，检查多个输入的设备和精度一致 |
| `backends/cpu.py` | CPUStorage、分配与复制、实数运算、训练更新，以及量子门和读出内核 |
| `core/tensor.py`、`core/grad.py` | 包装存储、记录计算关系和传播梯度；device/dtype 从存储读取 |
| `core/operations/` | 形状规则和局部反向公式，实际数值计算调用命名的后端运算 |

前向结果、梯度、SGD/AdamW 状态、BatchNorm 统计、KV 缓存和量子伴随向量都保留在存储中。普通数字和列表在进入计算时导入，`tolist()`、`item()`、采样、文件保存与可视化是明确的数值读取出口。当前 LM 仍使用 Python 随机采样，只读取最后一个位置的得分。

从普通数据创建 Tensor 会复制输入；后端产生的 Storage 可直接包装为 Tensor，不再绕过列表重新展开。reshape、切片与 transpose 维持复制语义，叶子梯度各自独立。CPU 下 `tensor.values` 和 `state.amplitudes` 保留实时调试入口，实际返回可变 CPUStorage，而非 Python list；它们不是列表快照，修改仍作用于原数据。显式克隆存储使用 `storage.copy()`，读取独立普通数据使用 `tolist()`。模型与优化器 JSON 格式保持不变。

当前只实现 CPU；Tensor 可显式指定 `device="cpu"`、`dtype="float64"`，State 可指定 `device="cpu"`。不支持的设备或精度明确报错。CUDA、实数 float32 和共享视图随后实现；连续存储本身不会自动加速 Python 循环，通用实数运算核也不承诺支持复数 Tensor。

## 最小反向传播

```python
from qutai import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss = (x + x).sum()
loss.backward()
print(x.grad.tolist())  # [2.0, 2.0, 2.0]
```

数学接口支持实数一阶反向传播：`add`、`subtract`、`multiply`、`divide`、`negate`、`square`、`relu`、`gelu`、`exp`、`log`、`sum`、`mean`、`reshape`、`transpose`、`stack`、`matmul`、`conv2d` 和 `max_pool2d`。支持二元广播、单轴归约、负轴、keepdims、任意维轴置换和同形状的显式输出梯度。非单元素结果需要传入输出梯度，例如 `result.backward([1, 2, 3])`。

`qutai.exp(data)` 计算以 e 为底的指数，`qutai.log(data)` 计算自然对数，后者要求输入大于零；二者保留输入形状。`qutai.argmax(data, axis=None, keepdims=False)` 返回最大值第一次出现的整数编号；不指定轴时按一维展开后的顺序编号。即使输入为 Tensor，`argmax` 也返回普通整数或整数嵌套列表，不参与求导。

一元运算的前向与局部反向函数在 `elementwise.py` 中成对排列：平方传递 `g * 2x`，取负传递 `-g`，ReLU 仅在输入为正的位置传递梯度，零点约定取零。

反向入口统一为 `loss.backward()`：Tensor 方法负责准备输出梯度，并将计算得到的叶子梯度保存为 Tensor；`grad.backward` 只沿计算图传播梯度存储，不导入或创建 Tensor。`types.py` 中的 `GradNode` 描述传播所需的三个字段，仅供类型检查，Tensor 不需要继承它。Tensor 调用 grad，grad 使用节点协议和 context 中的记录状态，所有 import 都在模块顶部。`storage` 保存一维数值，`parents` 与 `backward_rule` 保存计算关系，供内部模块直接协作；CPU 的 `values` 仅用于调试，`tolist()` 返回独立的普通数据。前向与反向之间不要修改这些数据或形状。

局部反向规则与前向计算放在同一个运算文件，`core/grad.py` 负责统一传播；CPU 的 `unbroadcast` 将广播梯度求和还原。新增运算如果尚未登记反向规则，求导时会明确报错，避免漏算贡献。

只有叶子 Tensor 保存 `.grad`，多次反向的结果会累加，`zero_grad()` 清除梯度。当前计算图随结果对象保留，允许对同一图重复反向；不支持高阶导数。不要在前向和反向之间修改输入数据或形状。

## 线性层

```python
from qutai import nn

layer = nn.Linear(2, 1, seed=42)
prediction = layer([[1, 2]])
optimizer = nn.SGD(layer, learning_rate=0.1)
```

`Linear(in_features, out_features, seed=None)` 变换输入的最后一轴，支持向量、二维批次和高维序列，例如 `(批次, 长度, in_features)` 变为 `(批次, 长度, out_features)`。权重形状为 `(in_features, out_features)`，计算公式是 `输入 @ 权重 + 偏置`，返回 Tensor。输入和输出特征数须大于零。

权重使用局部随机数生成器，在 `±1 / sqrt(in_features)` 范围内均匀初始化，偏置从零开始；固定 seed 可以复现初始化，不改变全局随机数状态。`weight`、`bias` 都是开启梯度记录的 Tensor，`parameters()` 返回它们的引用。

## 卷积、池化与展平

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, seed=None)` 接收 `(批次, 通道, 高, 宽)` 数据。权重形状为 `(输出通道, 输入通道, 核高, 核宽)`，使用适合 ReLU 的 He 均匀初始化；偏置从零开始，`bias=False` 时只提供权重参数。卷积核、步长和填充都支持整数或 `(高, 宽)` 二元组，窗口按向下取整确定输出大小。

`nn.MaxPool2d(kernel_size, stride=None, padding=0)` 按窗口取最大值，默认步长等于窗口大小。填充不参与最大值比较，因此不会让负数窗口错误地选中零；并列时选第一个最大值，反向只向选中位置传递梯度，重叠窗口的贡献相加。填充每侧不超过对应窗口尺寸的一半。

`nn.Flatten(start_dim=1)` 从指定轴展平至末尾，默认保留批次轴，例如 `(32, 16, 5, 5)` 变成 `(32, 400)`；它复用 `reshape` 的反向规则。池化和展平没有可训练参数，这三种层都返回 Tensor。

函数入口为 `qutai.conv2d(data, weight, bias=None, stride=1, padding=0)` 和 `qutai.max_pool2d(data, kernel_size, stride=None, padding=0)`；普通数据输入返回嵌套列表，有 Tensor 输入则保留 Tensor。卷积、池化的前向和局部反向规则一起放在 `core/operations/convolution.py`，纯数值循环在 `backends/cpu.py`。

## 多层模型

```python
from qutai import nn

model = nn.Model([
    nn.Linear(1, 16, seed=42),
    nn.ReLU(),
    nn.Linear(16, 1, seed=43),
])
optimizer = nn.SGD(model, learning_rate=0.1)
loss_function = nn.MSE()
targets = [[0.25]]
prediction = model([[0.5]], train=True)
loss = loss_function(prediction, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
prediction = model([[0.5]], train=False)
```

`ReLU` 复用已有的逐元素运算和反向规则，接收普通数据或 Tensor，返回 Tensor，没有可训练参数。`Model(layers)` 按顺序执行各层，支持嵌套 Model；`parameters()` 收集参数引用并去重，共享参数只出现一次。自定义对象提供 `__call__` 和 `parameters()` 就能放进 Model 或交给 SGD；自定义层继承 `nn.Layer` 后，还能使用层的 `+` 组合。

Layer 默认提供空的参数集合；有可训练参数的自定义层重写 `parameters()`，返回对应 Tensor 的引用。

默认 `train=True`，BatchNorm 使用当前批次的统计量并更新运行统计，计算图按输入和参数的 `requires_grad` 记录。`train=False` 使用 BatchNorm 已有的运行统计，并在整个模型调用范围内暂停建图，返回不记录梯度的 Tensor，保留参数、输入及已有梯度的状态。嵌套模型不会重新开启外层已暂停的记录，正常结束或抛出异常都会恢复调用前的状态。之后可继续使用 `train=True` 训练。

## 层的相加与残差块

`main + shortcut` 会创建一个 `Residual`：两路接收同一个输入，各自计算后将输出逐元素相加。两路输出形状必须完全一致，不进行广播。`nn.Identity()` 原样传递输入，可以作为恒等捷径：

```python
main = nn.Model([nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)])
residual = main + nn.Identity()
result = residual([[1, 2, 3, 4]])
```

`Residual` 只负责相加。`nn.ResNetBlock(in_channels, out_channels, *, stride=1, seed=None)` 封装 `Conv → BatchNorm → ReLU → Conv → BatchNorm` 主分支，与捷径相加后再接 ReLU，整块已经包含末尾激活。通道或尺寸变化时，捷径使用 `1×1 Conv → BatchNorm` 投影；其余情况使用 Identity。ResNetBlock 可以直接逐块列在 Model 中，完整例子见 `examples/resnet.ipynb`。

## 保存与加载

```python
model.save("model.json")

restored = nn.Model().load("model.json")
prediction = restored([[0.5]], train=False)
```

文件使用 JSON，包含模型结构和参数表。结构记录各层的类型、顺序、配置及参数编号；参数表保存形状、数值与可训练标记。`Model().load(path)` 会自动恢复 `Linear`、`Conv2d`、`MaxPool2d`、`Flatten`、`ReLU`、`Softmax` 和嵌套 `Model`，共享参数按编号恢复为同一个 Tensor。`load()` 返回模型自身，也可以先创建模型再单独调用它。

模型已有层时，同一个 `load(path)` 方法会核对结构和参数形状，原地更新参数、保留参数对象与原有可训练标记，并清空旧梯度，已有优化器仍可继续使用。旧版仅含参数的 JSON 需要先创建同结构模型，再用这个方法载入，之后可重新保存为完整格式。空模型自动恢复需要包含结构的文件。

层的配置也随模型保存，包括卷积核尺寸、步长、填充、可选偏置、Flatten 的起始轴和 Softmax 的 axis。自定义层的计算接口不变；新增可保存的层时，需要补上对应的结构描述与重建规则。

`examples/regression.ipynb` 的最后一部分会保存平方模型，通过空模型的 `load()` 恢复，并对照保存前与加载后的预测结果。保存加载的 Notebook 演示统一放在这里，其余 Notebook 专注各自的训练、预测和可视化。

## 损失函数

先用 `loss_function = nn.MSE()` 创建均方误差对象，再调用 `loss = loss_function(prediction, targets)`。输入可以是数字、列表、元组或 Tensor，返回标量 Tensor，对全部元素的平方误差取平均。预测与目标必须具有相同形状，不进行隐式广播。计算组合已有的减法、平方和平均值；当 Tensor 输入开启梯度记录时，结果保留计算关系，支持 `loss.backward()`。两边均为普通数据时，结果不记录梯度。

先用 `loss_function = nn.CrossEntropy()` 创建交叉熵对象，再调用 `loss = loss_function(scores, targets)`。得分的最后一轴为类别，至少需要二维；目标是对应其余各轴的整数标签列表或元组。分类使用 `(样本数, 类别数)` 得分与 `[0, 2, 1]` 一类的标签；序列预测使用 `(批次, 长度, 词表大小)` 得分与 `(批次, 长度)` 的嵌套整数标签。类别编号从 0 开始，返回全部位置的平均损失，类型为标量 Tensor；仅对得分求导。样本数和类别数须大于零。计算直接使用减去最大得分后的 log-sum-exp，避免先转概率造成溢出或极小概率变成零。

损失对象可以在训练循环外创建并反复调用。对象不保存输入或计算图；需要求导时，计算关系记录在该次返回的 Tensor 中。

## 优化器

使用 `nn.SGD(model, learning_rate=0.1)` 为整个模型创建优化器，也可以传入单层，如 `nn.SGD(layer)`，或显式的 Tensor 参数集合。学习率默认为 `0.1`。传入模型或层时，优化器在内部调用已有的 `parameters()` 接口，无需手动调用；参数由层明确声明，例如 Linear 的 `weight`、`bias`，不会扫描对象的其他属性。

优化器持有参数引用，重复参数只保留一次；`zero_grad()` 清除梯度，`step()` 按 `参数 - 学习率 × 梯度` 更新数值，跳过没有梯度的参数。每次 `step()` 后梯度仍保留，需要清除时显式调用 `zero_grad()`。

`nn.SGD(model, learning_rate=0.05, momentum=0.9, weight_decay=1e-4)` 可以开启动量和权重衰减，二者默认均为 0，保留基础梯度下降的行为。

`nn.AdamW(model, learning_rate=0.001)` 同样接收模型、单层或参数集合。默认 `betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`：分别维护梯度与梯度平方的移动平均，按每个参数实际更新的次数进行偏差修正，再自适应更新参数。权重衰减直接作用于参数，不混入移动平均；算法可对照 [AdamW 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW)。没有梯度的参数不更新，也不计步或衰减。

梯度裁剪放在反向与参数更新之间，`nn.clip_grad_norm(model, max_norm=1.0)` 返回裁剪前的全局 L2 范数。如果超过上限，全部梯度按同一比例缩放；共享参数只计一次。它也接收单层或参数集合，不创建计算图，不修改参数值。上限为零时清零已有梯度；空参数或没有梯度时返回零，非有限梯度会明确报错。

```python
optimizer = nn.AdamW(model, learning_rate=0.001)
optimizer.zero_grad()
loss = loss_function(model(inputs), targets)
loss.backward()
gradient_norm = nn.clip_grad_norm(model, max_norm=1.0)
optimizer.step()
```

SGD 和 AdamW 都支持优化器状态保存与加载：

```python
model.save("model.json")
optimizer.save("optimizer.json")

model = nn.Model().load("model.json")
optimizer = nn.AdamW(model).load("optimizer.json")
```

优化器文件包含类型、超参数、参数形状及逐参数的更新状态：SGD 保存动量，AdamW 保存计步和两种移动平均，没有 version 字段。`load()` 恢复保存时的超参数与状态，返回自身，保留当前参数对象、参数值和已有梯度。恢复后继续正常执行 `zero_grad → backward → clip → step`。

模型与优化器应在同一次更新后配对保存。优化器状态按去重后的参数顺序对应，因此恢复时须使用同一模型结构和参数顺序；类型、数量或形状不一致会报错。文件不保存数据流位置或随机数状态，恢复数值后需要继续提供相同批次才能复现连续训练。


## 回归实验 Notebook

在 PyCharm 打开 `examples/regression.ipynb`，选择项目的 Python 3.14 环境，然后从上到下运行各个单元格。Notebook 合并了线性回归和拟合 `y = x²` 两个实验，逐步展示数据、模型、训练过程与预测结果，代码只使用标准库和 qutai。

最后一部分保存平方模型，在当前工作目录生成 `regression.json`，再通过 `nn.Model().load(...)` 恢复，并检查加载前后的预测完全一致。Notebook 包含已运行的输出，也可以直接阅读结果。

线性回归使用 64 个样本学习 `y = 2x₁ - 3x₂ + 0.5`，模型为单个 `Linear(2, 1)`，学习率 0.1，训练 301 步；最终权重应接近 `[[2], [-3]]`，偏置接近 `0.5`。

非线性回归使用 `Linear(1, 16) → ReLU → Linear(16, 1)`，用 65 个均匀分布在 `[-1, 1]` 的训练点学习 `y = x²`，学习率 0.1，训练 1001 步。两个实验都使用 `nn.MSE()`，并展示训练后的预测和真实目标。

`matmul` 现已支持向量和批量矩阵乘法；复数留待后续阶段。

## 阶段实验：三分类

在 PyCharm 打开 `examples/classification.ipynb`，选择项目的 Python 3.14 环境，再从上到下运行单元格。实验使用 [UCI Iris 数据集](https://archive.ics.uci.edu/dataset/53/iris)：150 朵鸢尾花，每朵记录萼片长度、萼片宽度、花瓣长度、花瓣宽度，单位均为厘米。类别编号 `0`、`1`、`2` 分别对应 `setosa`、`versicolor`、`virginica`。

数据保存在 `examples/data/iris.csv`，使用标准库 `csv` 读取，运行时无需联网或新增 SDK 依赖。来源、许可和整理方式见 [数据说明](https://github.com/utai-utai/Qutai/blob/main/examples/data/README.md)。固定种子 42，每个品种取 40 个训练样本、10 个测试样本，共 120 个训练样本和 30 个测试样本。仅用训练集计算均值和标准差，测试样本与新输入使用同一组统计量进行标准化。

网络为 `Linear(4, 16) → ReLU → Linear(16, 3)`，使用学习率为 0.1 的 SGD 和交叉熵。最后一层输出三个类别的原始得分，训练时直接交给损失对象：

```python
from qutai import argmax, nn

loss_function = nn.CrossEntropy()
scores = model(inputs)
loss = loss_function(scores, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

scores = model(test_inputs, train=False)
predictions = argmax(scores, axis=-1)
softmax = nn.Softmax()
probabilities = softmax(scores)
```

`nn.Softmax(axis=-1)` 沿指定轴把得分转换为概率，输出始终为 Tensor，形状与输入相同。默认沿最后一轴，每组概率和为 1；支持高维、负轴和反向传播，没有可训练参数。同一轴上的 Softmax 不改变得分的大小顺序，所以取类别时直接调用根入口导入的 `argmax(scores, axis=-1)`；需要查看概率时，先创建 `softmax = nn.Softmax()`，再调用 `softmax(scores)`，也可以将该层放进 Model 保存。交叉熵接收原始得分，训练时无需预先调用 Softmax。

Notebook 展示训练前后的交叉熵、测试准确率，以及四项花朵测量值、正确品种和预测品种的对照。最后可以修改四项测量值，查看新样本的类别概率和品种名称。

本次固定种子 42 分层划分为 120 个训练样本、30 个测试样本，训练 500 步后，交叉熵从 `1.13027901` 降至 `0.0729983676`，测试预测正确 `30/30`，准确率为 `100%`。这是本次小测试集划分的结果，不代表其他数据或划分上的通用表现。

## 阶段实验：MNIST 手写数字

在 PyCharm 打开 `examples/lenet.ipynb`，使用项目的 Python 3.14 环境，从上到下运行。示例读取真实 MNIST 图像，展示一张手写数字、各层实际形状、分批训练和独立测试集预测。

图像形状统一为 `(批次, 通道, 高, 宽)`，即 NCHW。网络采用 ReLU 和最大池化的 LeNet 教学版：`Conv2d(1, 6, 5, padding=2) → ReLU → MaxPool2d(2) → Conv2d(6, 16, 5) → ReLU → MaxPool2d(2) → Flatten → Linear(400, 120) → ReLU → Linear(120, 84) → ReLU → Linear(84, 10)`。训练使用 `nn.CrossEntropy()` 与 `nn.SGD(model, learning_rate=0.1)`，预测类别直接取原始得分的 `argmax`。

MNIST 直接使用官方的原始 IDX 压缩文件，缓存在 `examples/data/mnist/`。`load_mnist()` 已有缓存时直接读取，缺失时自动下载；来源及格式见 [数据说明](https://github.com/utai-utai/Qutai/blob/main/examples/data/README.md)。每张图像包含 28×28 个原始像素，数值为 0–255。

数据读取与预处理保留在 SDK 外部。`examples/datasets.py` 的 `Images` 集合保存紧凑的像素字节；`images[i]` 只将这一张图片转换为带形状的 Tensor，并除以 255。`Stream` 接收列表或可索引的数据集对象，根据当前批第一条样本的形状组批，并检查同批其余样本形状一致。它负责子集、打乱和分批，文件解析、图片尺寸调整与归一化由外部完成。例如先读取数据，再取一批：

```python
from examples.datasets import load_mnist
from qutai import Stream

images, labels = load_mnist("train")
print(images[0].shape)  # (1, 28, 28)
data = Stream(images, labels, batch_size=32, seed=42, limit=2000, progress=False)
inputs, targets = next(iter(data))
print(inputs.shape, targets)
```

示例图片集合保持字节表示，只转换取出的样本。`Stream` 不再接收 `shape` 参数；同一批的形状不一致时需先在外部预处理。默认每批 32 项、`shuffle=True`、`progress=True`；`len(data)` 返回批次数，每次遍历从头开始并按需重新打乱。固定 seed 可以复现整段多轮训练的打乱顺序；局部随机状态会逐轮前进。

训练时设置 `progress=True`，每轮显示批次进度、百分比和用时。进度条由 `visualization/progress.py` 实现，只使用标准库；迭代体执行完毕后才计入完成数。测试和样本展示使用 `shuffle=False, progress=False`，保持顺序并关闭进度。

Notebook 默认使用官方训练集前 2000 张图像训练 5 轮，使用官方独立测试集前 500 张图像评估，每批 32 张。`limit` 先选定固定前缀子集，再打乱其训练顺序；不会每轮换一批评估样本。每轮打印最后一批的交叉熵，训练结束后统一评估测试集。这是验证纯 Python CPU 训练流程的小规模实验，不是全量 MNIST 成绩；可在配置单元格调整样本数量和轮数。

此前固定种子运行 5 轮后，测试预测正确 `471/500`，准确率 `94.20%`；首批训练样本的交叉熵从 `2.301837` 降至 `0.116331`。精简版保留模型和训练规模，本次只以少量批次验证，完整训练与测试单元的输出留空。

## Notebook 看图

`qutai.visualization` 提供统一的 `Image(pixels, shape, scale=6)`，接收 CHW 像素字节：MNIST 使用 `(1, 28, 28)`，CIFAR-10 使用 `(3, 32, 32)`。把图像对象放在单元格最后一行，Notebook 就会显示真实灰度或 RGB 图像；`scale` 只控制最近邻显示倍数，不改变训练数据。PNG 使用标准库 `struct`、`zlib` 编码，没有新增 SDK 依赖。

```python
from qutai.visualization import Image

Image(pixels, shape=(3, 32, 32), scale=6)
```

## 阶段实验：ResNet 与 CIFAR-10

在 PyCharm 打开 `examples/resnet.ipynb`。数据来自 [CIFAR-10 官方页面](https://cave.cs.toronto.edu/kriz/cifar.html)，直接读取官方 binary 压缩包，保留原始 32×32 RGB 图像。`load_cifar10()` 与 MNIST 的读取接口一致，返回图片集合和标签字节；`images[0].shape` 为 `(3, 32, 32)`，分批时自动获取形状。原始压缩包缓存在 `examples/data/cifar10/`，来源和格式见 [数据说明](https://github.com/utai-utai/Qutai/blob/main/examples/data/README.md)。

网络采用 CIFAR 版 ResNet-20 的三阶段结构，Notebook 显式列出各阶段的三个 ResNetBlock，第二、第三阶段的第一块设置 `stride=2` 下采样。每块已经包含相加后的 ReLU。通道或尺寸变化时，捷径使用 `1×1 Conv → BatchNorm` 投影，属于投影捷径变体；[原论文 CIFAR 实验](https://arxiv.org/html/1512.03385v1#S4.SS2) 使用的是 Option A 恒等及补零捷径。`nn.GlobalAvgPool2d()` 将 NCHW 特征转换为 `(批次, 通道)`，最后用 Linear 输出十类原始得分。

Notebook 默认 `width=4`，三个阶段使用 4、8、16 个通道，用于纯 Python CPU 的小规模流程验证；将 `width` 改为 16 可使用标准的 16、32、64 通道。默认选取 500 个训练样本、100 个官方独立测试样本，训练 3 轮，每批 16 张，学习率 0.05、动量 0.9、权重衰减 0.0001；仅按批将像素除以 255。这个实验不代表标准宽度网络或全量 CIFAR-10 的训练成绩。

每轮打印最后一批的训练损失，训练完成后评估独立测试集。`model(inputs, train=True)` 更新 BatchNorm 运行统计，`train=False` 使用已有统计进行预测并暂停记录计算图。

此前使用 ResNetBlock 写法完整训练 3 轮后，整轮平均训练交叉熵从 `2.35443` 降至 `2.05887`，独立测试正确 `19/100`，准确率 `19%`。这些是此前的训练记录。精简版保留相同模型和训练规模，本次只以少量批次验证，完整训练与测试单元的输出留空；当前样本量和轮数用于验证训练流程，识别效果仍有限。


## Tensor 的自然语法与索引

`x + y`、`x - y`、`x * y`、`x / y`、`-x` 和 `x @ y` 复用已有运算与反向规则，两侧都可混用数字、普通列表和 Tensor。`sum`、`mean`、`reshape`、`transpose`、`square`、`exp`、`log`、`argmax` 也提供同名 Tensor 方法，函数形式统一从 `qutai` 根入口导入。

```python
x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
loss = (x[:, 1:] * 2).sum()
loss.backward()
print(x.grad.tolist())  # [[0.0, 2.0, 2.0], [0.0, 2.0, 2.0]]
```

索引支持整数、切片（含负步长）、轴元组、`...` 和 `None` 新增轴；读取返回 Tensor。反向时未选位置为零，多处使用的贡献会累加。切片当前创建独立数据，不提供索引赋值、布尔索引或高级索引。`len(x)` 返回第一轴长度，`x.ndim` 返回维数；标量没有长度。

`x.reshape((2, 3))` 使用形状元组，`x.transpose((0, 2, 1))` 指定完整轴顺序。`x.T` 反转全部轴；三维及更高维的批量矩阵通常需要显式交换末两轴。

`matmul` 按最后两轴相乘，只广播前面的批次轴；一维向量临时升维后再还原输出。例如 `(B, T, D) @ (D, H)` 返回 `(B, T, H)`；两个向量相乘返回标量。梯度自动归约回共享权重或广播输入的原始形状。

Tensor 方法调用运算文件，再通过统一的 `_result` 创建结果并登记计算图。运算文件只读取 `Operand` 协议描述的属性，返回 `Computation`，即 `(values, shape, backward_rule)`，不导入或创建 Tensor；`argmax` 直接返回整数结果。依赖顺序为 `tensor → operations → cpu`，所有 import 都在文件顶部，根入口只导出公共接口。

## 单头 Attention 实验

`nn.Attention(in_features, out_features=None, seed=None)` 接收 `(批次, 序列长度, 特征数)`，通过 `query`、`key`、`value` 三个 Linear 投影，计算 `softmax(Q @ Kᵀ / sqrt(特征数)) @ V`。`weights(inputs)` 返回 `(批次, 查询位置, 来源位置)` 的权重，每行和为 1。当前实现为单头、无因果遮罩的自注意力，适合先理解序列之间的信息组合；所有梯度由已有 Linear、matmul、除法和 Softmax 组合产生。

`examples/attention.ipynb` 让网络读取四个 token 中带标记的数值，展示 Q/K/V 形状、结构图、训练曲线、注意力热图和独立测试预测。64 条训练序列训练 400 步，另用独立种子的 64 条序列测试；测试 MSE 约 `0.00018`，直接平均四个值的基线约 `0.2343`。最后通过 `model.save` 和 `nn.Model().load` 检查全部 token 输出严格一致。

## 统一可视化

`src/qutai/visualization/` 只使用标准库，面向普通 Python 数据，同时支持经典与量子模块：

| 文件 | 接口 | 用途与导出格式 |
| --- | --- | --- |
| `plots.py` | `Plot`、`Heatmap` | 折线、散点、坐标轴、注意力权重，导出 SVG |
| `tables.py` | `Table`、`Table.from_csv(path, limit=...)` | 普通行数据和 CSV 表格，导出 HTML |
| `images.py` | `Image` | CHW 灰度/RGB 字节图像，导出 PNG |
| `diagrams.py` | `Diagram`、`Network` | 通用有向图与模型结构图，导出 SVG |
| `quantum.py` | `Circuit`、`Probabilities` | 量子电路图和计算基概率柱状图，导出 SVG |
| `_svg.py` | 内部 SVG 输出工具 | 共享文字转义、画布和保存逻辑 |

```python
from qutai.visualization import Plot, Table, Image, Network

Plot(title="平方", xlabel="x", ylabel="y").line([-1, 0, 1], [1, 0, 1], label="x²")
Table.from_csv("examples/data/iris.csv", limit=8)
Network(model)
```

Notebook 把可视化对象放在单元格最后一行即可显示。每种对象都能 `.save(path)` 保存为独立文件，例如 `plot.save("curve.svg")`；无需绘图库、显示库或额外服务。`Network` 只读取结构，不执行前向计算：一般 Residual 展开两路输入与相加，ResNetBlock 折叠显示，Attention 的详细 QKV 连接可用通用 `Diagram` 表示。

回归 notebook 新增固定另一输入的线性回归坐标图和 x² 目标/拟合曲线；Iris 新增 CSV 与预测表；LeNet、ResNet 使用同一 `Image` 显示真实图像，并新增网络结构图。原 `examples/images.py` 已迁入 SDK。


## 通用 Transformer

Transformer 的共享组件放在 `nn/layers.py`，供图像和语言模型复用：

| 接口 | 功能 |
| --- | --- |
| `LayerNorm(features, eps=1e-5)` | 对最后一轴独立归一化，含可学习的缩放和平移；训练与预测使用相同计算 |
| `GELU()` | 精确的 `x * (1 + erf(x / sqrt(2))) / 2` 激活 |
| `MultiHeadAttention(features, heads=1, causal=False, seed=None)` | 多头自注意力，合并各头后做输出投影 |
| `FeedForward(features, hidden_features=None, seed=None)` | 每个位置独立执行 Linear → GELU → Linear，默认隐藏特征数为输入的四倍 |
| `TransformerBlock(features, heads=1, hidden_features=None, causal=False, eps=1e-5, seed=None)` | Pre-Norm 注意力和前馈网络，各自带一次残差相加 |
| `PatchEmbedding(in_channels, features, patch_size, seed=None)` | 通过不重叠卷积将 NCHW 图片投影为 patch 序列 |
| `PositionEmbedding(length, features, seed=None)` | 将可学习的位置向量加到输入，支持不超过预设长度的序列 |
| `Mean(axis=1, keepdims=False)` | 将指定轴汇总为平均值，ViT 用它汇总 patch 序列 |

```python
from qutai import nn

block = nn.TransformerBlock(16, heads=2, hidden_features=32)
```

TransformerBlock 的输入、输出均为 `(批次, 长度, 特征数)`，前向过程是：

```text
x = x + attention(attention_norm(x))
x = x + feed_forward(feed_forward_norm(x))
```

MultiHeadAttention 的 `features` 必须能被 `heads` 整除。`weights(inputs)` 返回 `(批次, 头数, 查询位置, 来源位置)`；`causal=True` 时，每个位置只关注自己和之前的位置，未来位置的权重与相应梯度均为零。默认使用不带遮罩的自注意力，适合图像分类。当前实现没有交叉注意力或 padding mask。

此前的 `Attention` 仍保留原单头教学接口和三个 Q/K/V 投影，旧 notebook 与模型文件继续可用。新的 MultiHeadAttention 另含输出投影。种子继续默认为 None；日常使用无需传入，示例才显式固定种子。内部随机对象采用 `random` 变量名。

LayerNorm 的前向和局部反向位于 `core/operations/normalization.py`，纯数字计算在 `backends/cpu.py`；它沿最后一轴使用总体方差，反向保存前向快照。GELU 的前向与反向位于 `elementwise.py`，也可通过 `qutai.gelu` 使用。多头拆分、合并与两次残差复用现有 reshape、transpose、matmul、Softmax 和广播的反向规则。

新增组件全部支持模型结构和参数的保存、空模型重建及原地加载，共享参数继续按编号保存。

## 阶段实验：ViT 与 MNIST

`examples/vit.ipynb` 使用真实 MNIST 的前四张训练图像和独立测试集的前四张图像。默认只更新参数两次，用于检查训练流程；可修改 `steps`，后续训练规模与 GPU 后端开发分开推进。

模型显式列出六层：

```python
model = nn.Model([
    nn.PatchEmbedding(1, 8, patch_size=7, seed=42),
    nn.PositionEmbedding(16, 8, seed=43),
    nn.TransformerBlock(8, heads=2, hidden_features=16, seed=44),
    nn.LayerNorm(8),
    nn.Mean(axis=1),
    nn.Linear(8, 10, seed=45),
])
```

28×28 图片变为 4×4 共 16 个 patch，每个 patch 有 8 个特征。Transformer 保持 `(批次, 16, 8)` 形状，序列平均得到 `(批次, 8)`，分类头输出 10 个得分。这是使用平均汇总的小型 ViT 教学结构，没有额外分类 token。

本次两步运行检查所有梯度与参数为有限数值，23 个参数 Tensor 全部发生更新；训练批交叉熵从 `2.07792774` 变为 `1.98971905`。这些输出用于确认程序可训练，不代表识别能力已训练完成。

Notebook 包含真实图像、网络结构图、TransformerBlock 的两次残差流程图、逐层形状、测试预测表，以及第一个头的注意力热图（前四个查询位置对全部 16 个 patch）。

开发顺序：通用 Transformer → ViT → LM → diffusion / DiT / flow matching。当前共享组件、小型 ViT、字符语言模型，以及像素空间 DDPM / flow matching 流程均已实现。


## 字符级语言模型

`qutai.lm` 提供字符编码和一个小型自回归 Transformer，整数序列分批复用根包的 `Stream`。它复用 `nn` 的可训练层、交叉熵、优化器和既有自动求导，不引入新的计算引擎或第三方依赖。

```python
from qutai import Stream, lm, nn

text = "小猫在窗边看雨。小猫在院子看花。"
tokenizer = lm.Tokenizer(text)
model = lm.Model(tokenizer, context_length=8, features=8, heads=2)
data = Stream(tokenizer.encode(text), context_length=8, batch_size=2, shuffle=False)
inputs, targets = next(iter(data))
optimizer = nn.AdamW(model, learning_rate=0.01)
loss_function = nn.CrossEntropy()

optimizer.zero_grad()
loss = loss_function(model(inputs), targets)
loss.backward()
nn.clip_grad_norm(model, max_norm=1.0)
optimizer.step()

print(model.generate("小猫", length=16, temperature=0.8, top_k=5))
```

| 文件 | 职责 |
| --- | --- |
| `lm/tokenizer.py` | 字符按首次出现的顺序编号，提供 encode / decode；空格、换行和 Unicode 字符均保留 |
| `core/data.py` 的 `Stream` | 统一图像与文本分批；文本生成完整整数窗口，目标向后错一个字符 |
| `lm/model.py` | 组合字符嵌入与因果网络，提供调用、参数收集、逐字生成和保存加载入口 |
| `lm/serialization.py` | 在已有神经网络结构、参数表之外保存词表顺序与上下文长度 |
| `nn/layers.py` 的 `Embedding` | 将原始整数编号查表为浮点特征向量；同一编号使用同一行可训练参数 |
| `core/operations/transforms.py` | Embedding 查表运算，反向复用 gather/scatter 将重复编号的梯度累加回权重表 |

`Tokenizer(text).characters` 是按首次出现顺序去重的字符字符串，`.size` 是词表大小；`encode` 返回普通整数列表，未知字符会报中文错误。`decode` 按相同顺序还原文本，不接受负编号或浮点编号。

`Stream(tokens, context_length=16)` 的默认上下文长度为 16、每批 32 段；没有单独的 LM 分批接口。输入窗口不重叠，只保留包含下一字符目标的完整窗口，最后不足一批的窗口仍然输出。文本太短时没有批次。默认 `shuffle=True`，`shuffle=False` 保持窗口顺序；可选 seed 用于复现。`limit` 在文本模式下限制窗口数量。每段的目标等于原文本向后移动一个字符。

`lm.Model(tokenizer, context_length=16, features=16, heads=2, depth=1, hidden_features=None, seed=None)` 先通过 `.embedding` 接收二维整数编号，再由 `.network` 执行 PositionEmbedding、指定层数的因果 TransformerBlock、LayerNorm 和词表 Linear。编号保持原始整数，进入 Embedding 后才成为浮点 Tensor。`.layers` 只按真实顺序展示这些层，供结构图和参数保存使用。

`model(inputs)` 返回 `(批次, 长度, 词表大小)` 的原始得分，直接交给 `nn.CrossEntropy` 与嵌套整数 targets，不需要手动展开。`train=False` 暂停建图并使用预测状态，离开调用后恢复原有 context；SGD 和 AdamW 通过 `model.parameters()` 自动取得全部参数引用。

`generate(prompt, length=20, temperature=0.0, top_k=None, seed=None, use_cache=True)` 返回原开头加上指定数量的新字符。默认温度为零，沿用每步选最高分字符的贪心方式；正温度先按得分计算稳定的指数权重，再按概率抽取。温度越低越偏向高分字符，`top_k` 可限定最高分的 k 个候选，超过词表大小时使用全部字符；采样参数的含义也见 [生成参数说明](https://huggingface.co/docs/transformers/main_classes/text_generation)。seed 默认不固定，指定后可复现，不影响全局随机数。

生成始终使用预测状态，不清除已有梯度；超出上下文长度时只将最后一段编号送入网络，返回文本保留完整开头。开头至少包含一个词表中的字符，暂不使用额外的起止 token。

`generate` 默认启用 KV cache，`use_cache=False` 可关闭作对照。首次计算当前开头，之后只对新字符执行嵌入、Q/K/V 投影与前馈，已有 K/V 按每层调用位置独立缓存；位置编码从已处理长度继续，因果掩码覆盖旧缓存和当前片段。注意力仍需要读取历史 K/V。结构原理见 [KV cache 说明](https://huggingface.co/docs/transformers/cache_explanation)。

缓存只在一次续写中存在，不参与训练，不改动参数或已有梯度，也不写入模型文件。当前模型采用可学习的绝对位置，生成超过上下文窗口时，保留字符的位置和前文都会变化，因此重新计算最后一个完整窗口并重建缓存，保证与原来的滑窗计算一致。窗口未满时可以减少重复计算；窗口填满后每次移动都要重建。

内部 `nn/cache.py` 只负责单层 K/V 的数值存储与追加；`MultiHeadAttention`、`TransformerBlock` 的 `cache=` 用于推理，普通调用保持原训练和反向规则；`PositionEmbedding` 的 `offset=` 选择当前片段对应的位置。LM 的 `_cached` 按网络顺序传递这些信息，外部继续只调用 `model.generate(...)`。

语言模型 JSON 包含 `model`、`parameters`、`characters`、`context_length` 四项，没有 version 字段。词表顺序决定字符编号，加载时会一起恢复并核对网络输入、输出与上下文长度。使用 `lm.Model().load(path)` 自动恢复全部信息；已有模型调用 `.load(path)` 会核对词表、上下文和网络结构，然后原地恢复数值、清除旧梯度，保留参数对象与优化器引用。

## 阶段实验：逐字续写

`examples/language_model.ipynb` 使用 `examples/data/language.txt` 的 180 个原创中文字符，共 44 种字符，无需下载数据。Notebook 展示词表、下一字符目标、网络结构、各层形状、因果注意力热图和续写结果。

默认上下文为 8、每批 2 段、特征数 8、2 个注意力头和 1 个 TransformerBlock。示例固定 seed=42，使用 AdamW，严格只更新两步，每步将全局梯度范数限制为 1；最后对照贪心与 temperature=0.8、top_k=5 的随机续写，再比较启用与关闭 KV cache。长度 2 的开头续写 6 字时，每层 K/V 投影处理字符数从 27 降为 7，文本一致。这是训练与生成流程验证，不是语言质量结果；Notebook 不保存模型或优化器文件。


## 像素空间生成：DiT、DDPM 与 flow matching

`nn.DiT` 是接收图片和时间的精简网络，`generation.Diffusion` 与 `generation.FlowMatching` 分别提供训练目标和采样规则；`generation.Model` 将两者组合，沿用 `model.save(path)` / `generation.Model().load(path)` 的单一入口。

```python
from qutai import Stream, generation, nn

method = generation.Diffusion(steps=8)
network = nn.DiT(1, 28, patch_size=7, features=8, heads=2, hidden_features=16)
model = generation.Model(network, method=method)
optimizer = nn.SGD(model, learning_rate=0.001)
loss_function = nn.MSE()

# images 为 0–1 的可索引图片集合或 NCHW Tensor。
data = Stream(images, method=method, batch_size=2, limit=4)
for inputs, times, targets in data:
    optimizer.zero_grad()
    loss = loss_function(model(inputs, times), targets)
    loss.backward()
    optimizer.step()

samples = model.sample(count=1)
```

`generation.Model(network)` 默认选择 16 步 DDPM；传入 `method=generation.FlowMatching(steps=8)` 则使用直线条件路径与 Euler 采样。两种方法共用网络代码，按不同目标分别训练参数。`seed` 默认为 None，示例显式传入种子用于复现。

| 文件 | 功能 |
| --- | --- |
| `nn/dit.py` | TimeEmbedding、带条件缩放/平移/门控的 DiTBlock、图片分块与还原 |
| `nn/serialization.py` | 扩展新网络、条件块和时间编码的结构重建，复用既有参数编号与共享引用 |
| `generation/diffusion.py` | 共享高斯噪声；DDPM 的余弦日程、噪声预测与后验采样；Flow Matching 的直线插值、速度预测与 Euler 积分 |
| `generation/model.py` | 网络调用、参数收集、采样、轨迹与保存加载入口 |
| `core/data.py` | Stream 原生支持生成模式，自动准备输入、时间和目标 |
| `generation/serialization.py` | 保存网络、所选方法和干净图片范围，恢复后可继续训练或采样 |
| `visualization/images.py` | 浮点 Tensor 转 PNG，以及内嵌 PNG 的并排图像网格 |

`nn.DiT(in_channels=1, image_size=28, patch_size=7, features=16, heads=2, depth=1, hidden_features=None, seed=None)` 使用 NCHW 输入，`times` 是长度等于批次大小的 `[0,1]` 浮点时间列表。`model(images, times, train=...)` 返回同形状的原始预测，不在网络中限制取值范围。image_size 和 patch_size 都支持整数或高宽二元组，图片尺寸须能被 patch 整除。

DiT 保留所有图片 token，时间用固定正弦/余弦编码（内部放大到 1000 时间尺度）后经过可训练 MLP。每个双向注意力块由时间产生平移、缩放与残差门控；最后将 token 投影为像素块并拼回图像。首版在现有 LayerNorm / GELU 上组合时间调制，使用像素空间、仅时间条件和单倍图像通道输出，是精简的 DiT 风格结构。所有梯度复用现有运算，不增加另一套反向引擎。

调制末层和最终输出投影为零初始化。因此初始块保持恒等、初始预测为零；头几步部分参数的梯度或更新可以为零。本次两步实验中，全部 29 个参数 Tensor 都有有限梯度，9 个 Tensor 发生更新，总计 2569 个可训练数值。测试另外设置非零调制与输出，数值检查全部小网络参数和时间编码的梯度。

DDPM 训练时随机为每张图选 k=1…T，网络时间为 k/T，目标为所加高斯噪声。采样依次从 T 回到 1，使用后验方差，最后一步不再添加噪声。余弦日程与 beta 上限 0.999 来自 [Improved DDPM 的噪声日程](https://arxiv.org/html/2102.09672v1)；实际 alpha_bar 由裁剪后的 alpha 累计计算，保持训练与采样系数一致。

FlowMatching 令 `x_t=(1-t)*noise+t*images`，目标为 `images-noise`；采样从噪声 t=0 出发，按 `x += model(x,t)/steps` 积分到 t=1。时间方向由各方法管理，两者的采样循环各自实现。

`Stream(images, method=method)` 原生支持 Diffusion 和 FlowMatching，直接输出 `(inputs, times, targets)`。生成模式接收 0–1 图片，每批自动映射到 −1–1，再由所选生成方法准备时间、噪声和目标，无需传类别标签或回调函数。Stream 使用 `GenerationMethod` 协议，不导入 generation；算法公式保留在 `generation/diffusion.py`。`seed` 同时控制打乱、时间和噪声，局部随机状态会跨批次和轮次推进，不会每批重新固定。无标签且未指定 method 时，保留现有文本窗口行为。`model.sample(count=1, seed=None)` 只保留最终结果；`model.trajectory(...)` 返回初始噪声及每一步状态，共 steps+1 项。采样暂停建图和训练行为，不清除既有梯度；每个状态 yield 前恢复调用者的 context，中途停止迭代也不会泄漏状态。原始采样结果始终保留实数范围。

保存文件含 `model`、`parameters`、`generation` 和 `value_range`，不含 version 或随机状态。空模型 load 会自动恢复 DiT 与所选方法；已有模型会检查方法配置与网络结构，再原地恢复参数并清除旧梯度，保留参数对象、冻结选择和优化器引用。固定相同 seed 后，加载前后的预测和采样结果严格一致。

## 生成模型 Notebook 与图像展示

`examples/dit.ipynb` 复用本地 MNIST，演示 DiT 与 Flow Matching。Stream 取四张图片、每批两张，自动将当前批从 `[0,1]` 映射到 `[-1,1]`，再生成输入、时间和目标速度。将 method 改成 Diffusion 即可使用噪声预测训练与 DDPM 采样。训练循环遍历两批后结束，随后跑完整 8 步采样。

每批使用不同的图片，并重新采样时间和噪声。两步用于验证训练流程，尚不足以得到可识别的数字；原始采样数值保留真实范围，只在显示图片时裁剪。

```python
from qutai.visualization import Image, ImageCombine, Network

Image.from_tensor(samples[0], value_range=(-1, 1))
frames = model.trajectory(count=1, seed=42)
ImageCombine([Image.from_tensor(frame[0], scale=4) for frame in frames], columns=3)
Network(model)
```

`Image.from_tensor` 接收 CHW 浮点 Tensor，只在生成 PNG 显示像素时映射、裁剪和取整，不修改 Tensor 或计算图；原来的字节图像接口与 PNG 输出保持兼容。`ImageCombine` 用内嵌 base64 PNG 的 HTML 展示多张图像，也能 `.save("images.html")`，不依赖外部图片服务。Network 为 DiT 展示图片与时间两条输入分支。


## 量子基础与 Bell 态

```python
from qutai import quantum as q, visualization as vis

circuit = q.Circuit(2, [q.H(0), q.CNOT(0, 1)])
state = circuit.run()
print(state.probabilities())   # 约 [0.5, 0.0, 0.0, 0.5]
print(state.counts(1000))      # 仅出现 00 和 11，次数之和为 1000

# 在 Notebook 中，分别放到单元格最后一行即可显示。
vis.Circuit(circuit, title="Bell 电路")
vis.Probabilities(state, title="Bell 态概率")
```

量子比特从 0 开始编号，**q0 为位串最左侧的最高位**；电路图中 q0 是最上方的线。两比特的振幅按 `|00⟩、|01⟩、|10⟩、|11⟩` 排列。门列表按顺序执行，`CNOT(control, target)` 的第一个参数为控制位。

| 文件 | 功能 |
| --- | --- |
| `quantum/gates.py` | Gate 小矩阵、I/H/X/Y/Z/S/T、RX/RY/RZ、CNOT/CZ/SWAP |
| `quantum/observables.py` | 局部厄米观测量、不同比特的张量积、实数加权和 |
| `quantum/state.py` | 复数振幅、概率、独立采样、频数、测量坍缩与复制 |
| `quantum/circuit.py` | 组合量子门、run 执行、参数收集与可求导期望值及概率 |
| `quantum/grad.py` | 伴随求导与逐门参数移位，复用现有反向引擎 |
| `quantum/qaoa.py` | I/Z 成本的 QAOA 模板、MaxCut 成本构造与参数收集 |
| `backends/cpu.py` | apply_gate 在所选量子比特上应用局部矩阵 |
| `visualization/quantum.py` | 电路结构、概率分布与单比特 Bloch 球的 SVG 展示 |

`State(qubits)` 默认准备全零态；`State(qubits, amplitudes)` 接受长度为 `2**qubits` 的已归一化振幅列表，也可直接使用 Python 内置复数。`.amplitudes` 是复数存储，`.probabilities()` 返回各振幅的模平方。

`state.apply(q.H(0))` 原地修改状态并返回自身；`state.copy()` 得到独立副本。`Circuit(qubits, gates)` 保存门列表，`.append(gate)` 追加门；每次 `circuit.run()` 都从全零态重新执行，`circuit.run(state)` 则复制传入状态后再执行，不修改输入。

旋转门使用 `q.RX(wire, angle)`、`q.RY(wire, angle)`、`q.RZ(wire, angle)`，角度单位为弧度，遵循标准的半角旋转矩阵，定义可对照 [RX](https://docs.pennylane.ai/en/stable/code/api/pennylane.RX.html)、[RY](https://docs.pennylane.ai/en/stable/code/api/pennylane.RY.html)、[RZ](https://docs.pennylane.ai/en/stable/code/api/pennylane.RZ.html)。自定义 `Gate(name, wires, matrix)` 时，矩阵维度为 `2**len(wires)`，须为酉矩阵；矩阵的局部基底按 wires 指定的顺序排列。模拟时直接更新相关振幅，不构造整个状态维度的方阵。

三类读出操作有明确区别：

- `state.probabilities(wires=None)` 返回理论概率；指定 `wires=0` 或 `wires=(2, 0)` 时，对未选择的比特求和，结果按指定的比特顺序排列。
- `state.sample(shots=1, wires=None, seed=None)` 模拟重复准备同一状态后的独立测量，返回位串列表，不改变状态；`state.counts(shots=1024, ...)` 返回实际出现过的位串及次数。允许零次采样。
- `state.measure(wires=None, seed=None)` 测量一次，返回位串，并原地将状态坍缩到对应子空间后归一化；未测比特保留与结果相容的叠加。

概率、采样和坍缩测量均在计算基下进行；单比特结果也返回字符串 `"0"` 或 `"1"`。seed 默认不固定，显式传入时可复现采样，且不会改变全局随机数状态。量子态继续独立使用复数振幅，旋转角度和可求导期望值使用现有实数 Tensor。

`examples/bell.ipynb` 已从头执行，依次展示初始态、H 门后的叠加、Bell 电路、振幅与理论概率、边缘概率、1000 次采样以及部分测量坍缩。示例默认随机采样，只出现 `00` 和 `11`，次数之和为 1000；有限次统计无需恰好各占一半。电路和概率图均内嵌在 Notebook 中，不额外保存模型或图片文件。


## 可训练量子电路

```python
from qutai import Tensor, nn, quantum as qu

angle = Tensor(0.3, requires_grad=True)
circuit = qu.Circuit(1, [qu.RY(0, angle)])
optimizer = nn.SGD(circuit, learning_rate=0.5)
loss_function = nn.MSE()

for step in range(2):
    optimizer.zero_grad()
    prediction = circuit.expectation(qu.Z(0))
    loss = loss_function(prediction, 0.5)
    loss.backward()
    optimizer.step()
```

`RX`、`RY`、`RZ` 的 angle 可以是普通数字或仅含一个元素的 Tensor。门保留原始 Tensor 引用，每次读取 `.matrix` 或执行电路时生成当前角度的矩阵。`gate.bind()` 返回独立的数值副本，`gate.bind(angle)` 可为旋转门指定一个固定角度；可视化同样使用数值快照。

`state.expectation(observable)` 计算 `<psi|O|psi>` 并返回普通 float，不改变量子态。观测量支持固定的厄米 Gate，例如 `qu.X(0)`、`qu.Y(0)`、`qu.Z(0)`，也支持这些算符的张量积与实数加权和。`Hermitian(wires, matrix)` 接受一般局部厄米矩阵，不要求矩阵为酉矩阵。Tensor 角度或 Tensor 系数的观测量暂不支持。

训练通过 `circuit.expectation(observable, state=None)` 计算期望值：门角度均为普通数字时返回 float；存在 Tensor 角度时返回标量 Tensor，根据 `requires_grad` 和当前 recording 状态记录计算关系。传入非空观测量列表时，State 返回 float 列表，Circuit 返回 float 列表或形状为 `(观测量数量,)` 的 Tensor，顺序与输入一致。可选 state 是数值初态，默认使用全零态。`circuit.run()` 返回的 State 只负责数值模拟；训练读取期望值或概率时，直接使用电路的 expectation 或 probabilities 入口。

`circuit.parameters()` 返回门引用的 Tensor 角度并去重，保留原对象及其可训练标记，直接兼容 `nn.SGD(circuit, ...)`。同一角度用于多个门时，正反向都保留每个门的位置，梯度贡献由 `core.grad` 累加。经典计算生成的 Tensor 角度也能接回原计算图；如果角度来自 `2 * weight + bias` 这类表达式，应在每次前向中重新计算表达式并搭建对应电路。

量子反向默认使用伴随求导，`circuit.expectation(..., method="parameter-shift")` 可切换为参数移位。两种方法都对 RX/RY/RZ 的角度求导，并通过现有 `core.grad.record` / `backward` 继续统一传播；外部仍使用 `loss.backward()`。它们保存本次前向的数值快照，不修改当前参数，也不重新建立计算图。参数移位规则可对照 [官方说明](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.param_shift.html)。

`examples/quantum_training.ipynb` 已从头执行，只有一个 RY 门和一个可训练角度，目标是让 Z 期望值接近 0.5。严格执行两次 SGD 更新，MSE 为 `0.207331 → 0.165694 → 0.103664`，最终角度约 `0.605936`、期望值约 `0.821969`。Notebook 保留电路图、训练前后概率、参数记录和损失曲线，不保存模型或额外图片。


## 量子与经典混合层：qnn.PQC

```python
from qutai import nn, qnn

model = nn.Model([
    nn.Linear(2, 2),
    qnn.PQC(qubits=2, depth=2, entangler="cnot"),
    nn.Linear(2, 1),
])
optimizer = nn.SGD(model, learning_rate=0.1)
loss_function = nn.MSE()
```

`PQC(qubits, depth=2, entangler="cnot", seed=None)` 在创建时，为每个深度、每个比特的位置分别随机选择 RX、RY 或 RZ，并随机初始化对应角度。之后旋转门种类与相邻比特的连接保持固定，每次训练只更新角度。纠缠门支持 `"cnot"` 和 `"cz"`，不区分大小写；每层依次连接 `(0,1)、(1,2)…`，单比特电路省略纠缠门。

`.rotations` 是形状为 `(depth, qubits)` 的旋转门名称元组，`.weight` 是同形状的可训练 Tensor，总计 depth×qubits 个角度。seed 默认不固定；显式指定可复现该实例的门结构和初始角度，不影响全局随机数状态。首版只提供这份完整模板，不包含自定义门或替换位置的接口。

前向先用输入特征构造逐比特 RY 编码，再执行 depth 组旋转与纠缠。末尾固定执行逐比特 RY(pi/4)，然后读取 Z 期望值，相当于从原态读取 `(Z-X)/sqrt(2)` 特征。固定读出旋转不增加训练参数，让末尾 CZ 及相位信息能影响输出。

输入支持普通列表或 Tensor，形状为 `(..., qubits)`；每个样本独立模拟并共享同一权重，返回同形状 Tensor。每次前向重新建立输入与角度的计算关系，批次中各样本的权重梯度会累加。`.parameters()` 只返回自身的 weight，前面经典层生成的输入继续通过原计算图接收梯度，不会被误当作 PQC 的参数。空批次保持形状并传播零梯度。

`pqc(inputs, train=False)` 可以独立暂停建图；放进 `nn.Model` 后，`model(inputs, train=False)` 同样暂停整个网络建图，保留参数和已有梯度。量子部分复用 `quantum.Circuit.expectation` 的默认伴随求导与 `core.grad` 的统一传播。

新增的 `qutai.stack(data, axis=0)` 将同形状输入沿一个新增轴组合，支持负轴。有 Tensor 输入时返回 Tensor，普通输入返回普通数据。反向将输出梯度分回各输入，重复引用的输入会累加贡献；PQC 用多观测量入口读取每个样本的全部特征，再用 stack 组合样本，保留完整计算图。

```python
from qutai import stack

stack([[1, 2], [3, 4]])          # [[1.0, 2.0], [3.0, 4.0]]
stack([[1, 2], [3, 4]], axis=1)  # [[1.0, 3.0], [2.0, 4.0]]
```

PQC 的保存加载沿用完整模型的单一入口：

```python
model.save("hybrid.json")
restored = nn.Model().load("hybrid.json")
```

文件包含比特数、深度、纠缠门、实际旋转门序列及参数编号，不依赖重新随机生成原始布局。空模型恢复实际结构、参数共享与可训练标记；已有模型原地加载时继续检查完整结构（包括旋转门种类），保留参数对象、优化器引用和原有冻结选择，并清除旧梯度。

`PQC` 只依赖 core 和 quantum。`core.types.LayerLike` 描述所有层共同的调用与参数接口，NN 容器通过该接口接收 PQC，避免 qnn 与 nn 双向导入；已有经典层继续保留自己的 Layer 基类。

| 文件 | 新增功能 |
| --- | --- |
| `qnn/pqc.py` | 固定随机旋转、输入编码、CNOT/CZ、批次前向与参数收集 |
| `core/operations/transforms.py` | stack 数值计算及局部反向，复用已有 CPU transpose |
| `core/tensor.py` | stack 的输入与结果包装、计算关系登记 |
| `nn/serialization.py` | PQC 结构和参数恢复 |
| `visualization/diagrams.py` | 混合网络中的 PQC 结构标签 |

`examples/hybrid.ipynb` 已从头执行：九对输入学习 `y=0.5*x0-0.3*x1`，网络为 `Linear(2,2) → PQC(2,depth=2) → Linear(2,1)`。严格执行两次 SGD 更新，MSE 为 `0.218523 → 0.209098 → 0.202538`，前置 Linear、PQC、后置 Linear 的参数都获得有限梯度并发生更新。Notebook 保留混合网络图、具体样本电路、损失记录、曲线和预测表，不额外保存文件。


## 组合观测量、多结果读出与 VQE

```python
from qutai import Tensor, nn
from qutai.quantum import CNOT, RY, X, Z, Circuit

hamiltonian = -(X(0) @ X(1)) - 0.5 * Z(0) - 0.5 * Z(1)
angle = Tensor(0.2, requires_grad=True)
circuit = Circuit(2, [RY(0, angle), CNOT(0, 1)])
optimizer = nn.SGD(circuit, learning_rate=0.2)

features = circuit.expectation([Z(0), Z(1), X(0) @ X(1)])
energy = circuit.expectation(hamiltonian)
energy.backward()
optimizer.step()
```

`@` 只表示不同量子比特上的张量积，可以作用于不相邻的比特，拒绝重叠比特。`+`、`-`、实数乘除和 `sum(...)` 组成带实系数的观测量和；普通数字项表示该数字乘以恒等算符。门继续描述如何改变量子态，Observable 描述读取什么；组合保存门的当前比特位置和矩阵快照，之后修改原门不会改变组合结果。

`Hermitian((0,), ((2, 1j), (-1j, 3)))` 可以描述非酉的局部厄米观测量。计算逐项作用局部矩阵，不构造整个系统的稠密矩阵，也不把非酉观测量当作量子门执行。

多个观测量共享一次前向电路演化。伴随求导先按上游梯度合并观测量对终态的作用，再逆序遍历一次电路；参数移位则对每个 Tensor 旋转门计算加减 pi/2 的状态，所有读出共用这两个状态。共享参数的各个门位置分别求导，最后由 core.grad 累加；PQC 已接入这个入口。

VQE 将哈密顿量的期望值直接作为最小化目标，复用 `energy.backward()` 和 SGD，不需要额外损失函数或 VQE 模型类。示例原理参考 [PennyLane 的 VQE 介绍](https://pennylane.ai/demos/tutorial_vqe)。`examples/vqe.ipynb` 的两比特能量为 `-sin(angle)-cos(angle)`，理论最低值为 `-sqrt(2)`。严格两次更新后能量为 `-1.178736 → -1.285991 → -1.346182`，Notebook 保留电路、联合概率、能量曲线和局部 Bloch 球，不保存额外文件。


## 单比特 Bloch 球

```python
from qutai import visualization as vis
from qutai.quantum import CNOT, H, Circuit

state = Circuit(2, [H(0), CNOT(0, 1)]).run()
vis.Bloch(state, wire=0, title="Bell 态中的 q0")
```

`Bloch(state, wire=0, title="")` 根据所选比特的 X、Y、Z 期望值绘制 SVG，`.vector` 为三个坐标，`.length` 为向量长度。只读取数值并保存快照，不改变量子态；Notebook 可直接显示，也可以调用 `.save(path)` 导出 SVG。

纯态向量长度为 1；纠缠态中单个比特的约化态可能位于球内，Bell 态中的每个比特位于球心。Bloch 球不把短向量强行归一化，也不能单独展示多个比特的全部关联。定义可对照 [IBM 的量子态可视化说明](https://quantum.cloud.ibm.com/docs/en/guides/plot-quantum-states)。`examples/bell.ipynb` 增加了 H 门后局部纯态与 Bell 态局部混合态的对照图。


## QAOA 与 MaxCut

VQE 用所选的参数化电路降低给定哈密顿量的能量；QAOA 使用固定的交替结构，把组合优化成本编码进电路。它们共用成本期望、自动求导和经典优化器。QAOA 的结构参考 [原始论文](https://arxiv.org/abs/1411.4028)。

```python
from qutai import nn
from qutai.quantum import QAOA, maxcut

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
algorithm = QAOA(4, maxcut(edges), depth=1)
optimizer = nn.SGD(algorithm, learning_rate=0.1)

optimizer.zero_grad()
energy = algorithm.expectation()
energy.backward()
optimizer.step()

state = algorithm.run()
print(state.probabilities())
```

`maxcut(edges)` 接受普通的无向边列表，每条边贡献 `(Z(i) @ Z(j) - 1) / 2`。对于任意分组位串，成本等于切边数的负数；最小化成本就对应最大化切边数。定义可对照 [PennyLane 的 MaxCut 成本](https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.maxcut.html)。空边列表生成零成本；每个边列表项计一次，重复边会重复贡献。

`QAOA(qubits, cost, depth=1, seed=None)` 支持单比特 I/Z 因子的张量积及实数加权和，包含 Z、ZZ 和更长的 Z 字符串。可以手写带权成本；其他类型的成本和直接提供的多比特 Hermitian 矩阵暂不解析。构造时保存固定成本快照，`.cost` 返回副本，之后修改原观测量不会改变电路结构。

`.weight` 的形状是 `(depth, 2)`，每一行分别为 gamma、beta，默认随机初始化；seed 显式传入时可复现且不影响全局随机数。`.parameters()` 只返回这组权重，可直接交给 SGD。

电路从全体 H 门产生的等幅叠加态开始，每层先执行 `exp(-i gamma cost)`，再执行 `exp(-i beta sum(X))`。Z 字符串使用 CNOT 奇偶校验、RZ(2 × coefficient × gamma) 和逆校验实现，混合部分为逐比特 RX(2 × beta)。成本中的常数只对应全局相位，在门序列中省略，但仍保留在能量期望中。

`.circuit()` 用当前权重重新建立派生角度，返回可视化和继续组合的 Circuit；`.expectation()` 返回可求导的标量 Tensor，`.run()` 返回数值 State。gamma、beta 对应多个门时，各个门的角度梯度通过经典乘法与切片规则传回同一个 weight。

`visualization.Graph(nodes, edges, groups=None, title="")` 使用圆周布局绘制无向图，支持整数或字符串节点。groups 是 `{节点: 分组编号}`，跨组边显示为绿色；接口支持 Notebook SVG 显示和 `.save(path)`。

`examples/qaoa.ipynb` 已从头执行：四节点环、深度 1、seed 42、学习率 0.1，严格两次更新的期望切边数为 `1.421633 → 2.585731 → 2.935219`，理论最大切边数是 4。最终最可能的分组 `0101` 切开四条边，互补分组 `1010` 同样最优。期望切边数是概率加权平均，单次测量的切边数仍为整数；两步实验只验证流程，不保证一般问题都能找到最优解。Notebook 保留问题图、电路、训练曲线、概率、采样表和分组图，不保存额外文件。


## 伴随求导与参数移位对照

```python
loss = circuit.expectation(observable)  # 默认 method="adjoint"
loss.backward()

# 对同一电路和输入，用参数移位核对梯度。
for parameter in circuit.parameters(): parameter.zero_grad()
loss = circuit.expectation(observable, method="parameter-shift")
loss.backward()
```

伴随求导保存前向终态，先构造所有读出按上游梯度加权后的状态向量，再从最后一个门往前撤销。每个旋转门直接使用精确导数矩阵得到角度梯度，不必为这个门重跑完整电路。逆序过程中只保留固定数量的状态向量，不缓存所有中间态；共享参数和经典计算得到的角度继续交给 core.grad 传播。

`Gate.adjoint_matrix` 返回局部矩阵的共轭转置，即酉门的逆矩阵；RX/RY/RZ 的 `derivative_matrix` 使用精确恒等式 `dR(theta)/dtheta = R(theta + pi)/2`。固定门不提供角度导数，名称恰好叫 RX/RY/RZ 的自定义固定 Gate 仍按实际矩阵处理。原理参考 [伴随求导介绍](https://pennylane.ai/demos/tutorial_adjoint_diff) 与 [Jones、Gacon 的论文](https://arxiv.org/abs/2009.02823)。

选择 `method="parameter-shift"` 时保留原算法：每个 Tensor 旋转门的位置额外执行两遍完整电路。两种方法使用独立的前向数值快照，支持任意固定纯态初态、多观测量、共享角度和重复 backward。PQC、VQE、QAOA 的调用方式不变，默认使用伴随求导。

可在项目根目录运行 `python examples/adjoint_benchmark.py` 复现比较。它固定同一电路、参数和成本，每种方法各预热一次，再交替顺序各测三次并取中位数；计时包含期望值前向和 `loss.backward()`，不包含构造电路、清梯度或更新参数。以下为本次 Python 3.14.2、macOS arm64、连续 CPUStorage 的纯 Python 实测，耗时会随机器和运行负载变化。

| 比特数 | 深度 | 总门数 | 参数化门位置数 | 伴随前向＋反向 | 参数移位前向＋反向 | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 18 | 8 | 1.541 ms | 7.865 ms | 5.10× |
| 6 | 4 | 50 | 24 | 12.446 ms | 171.159 ms | 13.75× |
| 6 | 8 | 94 | 48 | 23.027 ms | 599.850 ms | 26.05× |

本轮改为连续存储后，纯 Python 复数元素的读取与封装有额外开销，以上绝对耗时高于原列表版；这次迁移统一存储与后端接口，没有带来 CPU 内核加速。

这三组电路两种方法的最大梯度绝对差为 `2.78e-16`。测试还与独立有限差分对照，覆盖复相位、非自逆门、非相邻连线、一般厄米观测量、多结果的上游梯度、完整混合网络和 QAOA；执行次数测试验证伴随反向随电路长度线性增长，不对墙钟加速比作硬断言。


## 可求导的概率读出

```python
from qutai import Tensor, nn
from qutai.quantum import CNOT, RY, Circuit

angle = Tensor(1.0, requires_grad=True)
circuit = Circuit(2, [RY(0, angle), CNOT(0, 1)])
loss_function = nn.MSE()

probabilities = circuit.probabilities()
loss = loss_function(probabilities, [0.15, 0, 0, 0.85])
loss.backward()
```

`circuit.probabilities(wires=None, state=None, method="adjoint")` 返回计算基概率，支持全部比特、单个比特或按指定顺序排列的多个比特。q0 对应位串最左侧；`wires=(2, 0)` 的结果依次对应 q2、q0 的 `00、01、10、11`，其余比特被求和。概率定义可对照 [PennyLane 的说明](https://docs.pennylane.ai/en/stable/code/api/pennylane.probs.html)。

所有门角度都是普通数字时返回 float 列表；存在 Tensor 角度时返回形状为 `(2**所选比特数,)` 的 Tensor，是否建图仍由 requires_grad 与 recording 决定。默认从全零态开始，可传入固定数值 State。`state.probabilities()` 始终返回普通列表，不建立计算图或改变原状态。

概率反向默认使用伴随法，也支持 `method="parameter-shift"`。伴随法将每个完整基态振幅乘以它对应概率结果的上游梯度，再共用已有逆序传播；边缘概率通过相同的比特索引映射传回。参数移位每次读取整组概率，按上游梯度加权汇总。两种方法都不构造逐结果的投影矩阵，支持共享参数、经典角度表达式与前向数值快照。

期望值与概率共用 `Circuit._snapshot` 保存初态、门和参数关系，量子反向共用 `_adjoint` 逆序遍历，避免复制另一套计算图。

`examples/quantum_probabilities.ipynb` 已从头执行：单个 RY 角度与 CNOT 学习 `[0.15, 0, 0, 0.85]`，严格两次 SGD 更新，MSE 为 `0.192294 → 0.126238 → 0.074218`，P(11) 为 `0.229849 → 0.347529 → 0.464726`。Notebook 保留电路、训练前后概率、损失曲线和预测表，不保存额外文件。这里训练的是精确理论概率，有限次采样继续通过 State 的 sample、counts 完成。
