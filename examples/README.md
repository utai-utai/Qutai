# Qutai 教程

先在项目根目录执行 `python -m pip install -e .`，再在 PyCharm 中用同一个 Python 环境打开 Notebook，**将 Notebook 工作目录设为项目根目录**。数据导入统一使用 `from examples.data.data import ...`，无需每本重复修改 sys.path。

第一格只需指定 `device = "apple"`。CPU 或 NVIDIA GPU 分别改成 `"cpu"`、`"cuda"`；实数与量子态的精度由设备自动选择，不用再设置 dtype。教程不固定随机种子，重新运行时初始化、批次顺序和结果会变化。

修改 SDK 源码后，请重启 Notebook 内核并从第一格重新运行；只重跑 import 不会加载已导入模块的新代码。

## 阅读顺序

| 目录 | 教程 | 这一页要学什么 |
| --- | --- | --- |
| nn | [回归](nn/regression.ipynb) | 从直线到 x²，最后保存和加载模型 |
| nn | [Iris](nn/classification.ipynb) | 三分类、交叉熵与预测概率 |
| nn | [LeNet](nn/lenet.ipynb) | 卷积、池化和 MNIST 数字识别 |
| nn | [ResNet](nn/resnet.ipynb) | 用 MNIST 理解残差连接 |
| lm | [Attention](lm/attention.ipynb) | 从序列中找出被标记的数值 |
| lm | [ViT](lm/vit.ipynb) | 把 MNIST 切成 patch 交给 Transformer |
| lm | [语言模型](lm/language_model.ipynb) | 字符编号、下一字预测与续写 |
| lm | [DiT](lm/dit.ipynb) | 图片 patch、时间与生成过程 |
| quantum | [Bell 态](quantum/bell.ipynb) | 门、电路、概率、采样和 Bloch 球 |
| quantum | [混合网络](quantum/hybrid.ipynb) | 把 PQC 当成 nn.Model 中的普通一层 |
| quantum | [VQE](quantum/vqe.ipynb) | 优化电路参数，让能量降低 |
| quantum | [QAOA](quantum/qaoa.ipynb) | 用量子电路求解一个小 MaxCut 问题 |
| quantum | [梯度方差](quantum/barren_plateau.ipynb) | 2–8 个比特时，全局 cost 的梯度方差如何变化 |

## 一套训练写法

有数据的教程都用同一个循环；DiT 的批次额外包含时间 `times`：

```python
for epoch in range(epochs):
    for inputs, targets in data:
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
    print(epoch + 1, loss.item())
```

`data` 是 `Stream(...)`，每次遍历就是一轮；`epochs` 控制轮数，`batch_size` 控制每批样本数。VQE、QAOA 没有数据集，直接每轮更新一次电路参数。

教学模型和数据量较小，用于理解完整流程；识别准确率、续写与生成效果随训练量和随机初始化变化。只有回归教程保存模型到当前工作目录的 `regression.json`；文件不存在时自动创建。其余教程结束于预测或图表。

## 数据

[数据读取](data/data.py) 集中处理 [Iris CSV](data/iris.csv) 和 MNIST；Notebook 只取数据、组织 Stream，不展开下载和解压代码。MNIST 首次自动下载，之后读取 `examples/data/mnist` 缓存。图片按需转成 Tensor，无需图像、数据集或绘图库。

来源：[Iris / Fisher 1936](https://doi.org/10.24432/C56C76)（CC BY 4.0，增加表头并简化品种名）、[MNIST](https://github.com/cvdfoundation/mnist)。
