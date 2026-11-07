# 扩散模型生成

[diffusion.ipynb](diffusion.ipynb) 使用真实MNIST图片、`Stream`、`nn.DiT` 与 `generation.Diffusion`：随机加噪 → 预测噪声 → 从高斯噪声逐步去噪。

默认训练2轮、每轮2批，采样8步，只验证生成闭环，生成图通常仍有大量噪声。把方法一行换成 `generation.FlowMatching(steps=8)`，即可对照直线插值与速度预测；数据流和训练结构相同。

工作目录可为项目根、`examples` 或 `examples/generation`。MNIST共用 `examples/data/mnist` 缓存，读取与来源见 [共享数据说明](../data/README.md)。
