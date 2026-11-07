# 神经网络：从回归到图像分类

按 [回归](regression.ipynb) → [Iris](iris.ipynb) → [LeNet](lenet.ipynb) → [ResNet](resnet.ipynb) 的顺序，逐步学习数据、模型、损失、反向与优化器。再看 [Attention](attention.ipynb) 的序列读出与 [ViT](vit.ipynb) 的图片分块。

Notebook默认CUDA/float32，CPU可修改第一格的设备。工作目录可为项目根、`examples` 或 `examples/nn`。图像读取器与数据来源见 [共享数据说明](../data/README.md)。

回归保存当前平方模型到工作目录下的 `runs/regression.json` 并核对恢复预测；其他Notebook结束于测试或图表。LeNet、ResNet保留独立测试集，默认小前缀训练用于教学，不把演示结果当完整数据集基准。
