# 示例数据

## Iris

`iris.csv` 用于 `examples/nn/iris.ipynb`，包含 150 条记录，每个品种各 50 条。数据来源为 [UCI Machine Learning Repository：Iris](https://archive.ics.uci.edu/dataset/53/iris)，采用 [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)。

引用：Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C56C76](https://doi.org/10.24432/C56C76)。

本地文件取自 [UCI 官方下载包](https://archive.ics.uci.edu/static/public/53/iris.zip) 中的 `bezdekIris.data`，整理时增加 CSV 表头、忽略空行，并去掉品种名称的 `Iris-` 前缀；原有记录顺序与测量数值保持不变。

| 字段 | 含义 |
| --- | --- |
| `sepal_length` | 萼片长度，厘米 |
| `sepal_width` | 萼片宽度，厘米 |
| `petal_length` | 花瓣长度，厘米 |
| `petal_width` | 花瓣宽度，厘米 |
| `species` | `setosa`、`versicolor` 或 `virginica` |

Notebook 将三个品种依次编码为 `0`、`1`、`2`。本地 CSV 保留原始测量值，训练集划分和标准化在 Notebook 中执行。

## MNIST

`examples/nn/lenet.ipynb` 通过 `examples/datasets.py` 读取 MNIST 手写数字。原数据有 60000 张训练图像、10000 张独立测试图像，图像均为 28×28 的单通道字节图像，标签为数字 0–9。

本地直接保留 MNIST 原始 IDX 压缩文件。`load_mnist("train")` 和 `load_mnist("test")` 自动选择对应的训练、测试文件；已有缓存时可离线读取，缺失时才下载，无需手动传入文件路径或转换数据格式。

下载使用 [CVDF 官方 MNIST 镜像说明](https://github.com/cvdfoundation/mnist/blob/master/README.md) 中列出的 Google Storage 地址，缓存到此目录的 `mnist/` 子目录：

- `train-images-idx3-ubyte.gz`：训练图像。
- `train-labels-idx1-ubyte.gz`：训练标签。
- `t10k-images-idx3-ubyte.gz`：官方测试图像。
- `t10k-labels-idx1-ubyte.gz`：官方测试标签。

`examples/datasets.py` 负责解压和解析 IDX，检查类型、28×28 形状、图像数量与标签数量，返回可索引的 `Images` 图片集合与标签字节。集合内部保留原有顺序和原始像素字节；`images[i]` 取样时才除以 255，并返回形状为 `(1, 28, 28)` 的 Tensor。`qutai.Stream` 选定子集、打乱并分批，自动获取当前批首个样本的形状，输出 `(批次, 1, 28, 28)` Tensor 和整数标签列表。训练代码只接收每批的 `(inputs, targets)`。

Notebook 使用训练、测试文件各自的固定前缀子集，不将训练图像混入测试数据。

## CIFAR-10

`examples/nn/resnet.ipynb` 使用 [CIFAR-10 官方数据](https://cave.cs.toronto.edu/kriz/cifar.html)，由 Alex Krizhevsky、Vinod Nair、Geoffrey Hinton 创建。原数据包含 50000 张训练图像和 10000 张独立测试图像，均为 32×32 RGB 图像。官方引用为 Alex Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*，详见官方页面的 Reference。

`load_cifar10("train")` 或 `load_cifar10("test")` 自动缓存官方 binary 版本的 `cifar-10-binary.tar.gz`，目录为 `cifar10/`。官方公布的 MD5 为 `c32a1d4ab5d03f1284b67883e8d87530`。训练集来自 `data_batch_1.bin` 到 `data_batch_5.bin`，测试集来自独立的 `test_batch.bin`；读取时只通过 `tarfile.extractfile` 访问指定成员，不将压缩包全部解压到磁盘。

默认通过 [TensorFlow 源码列出的数据镜像](https://android.googlesource.com/platform/external/tensorflow/+/922be6369cdf0f724eff91a0bd9bd1d779786611/tensorflow/lite/micro/tools/make/third_party_downloads.inc) 的 Google Storage HTTPS 地址下载，使用同一原始压缩文件。本次下载已核对上述官方 MD5。

每条 binary 记录由一个标签字节和 3072 个像素字节组成，像素依次为 R、G、B 三个 32×32 平面，每个平面按行保存。读取后保留为逐图 CHW 字节；`images[i]` 取样时除以 255，返回形状为 `(3, 32, 32)` 的 Tensor。`Stream(images, targets)` 自动获取形状并输出 NCHW Tensor 和整数标签列表。数据集文件格式的解析与预处理都留在示例代码中。

类别编号 0–9 依次对应 `airplane`、`automobile`、`bird`、`cat`、`deer`、`dog`、`frog`、`horse`、`ship`、`truck`，可通过 `CIFAR10_NAMES` 查询。`qutai.visualization` 的 `Image` 与 MNIST 共用同一接口；这里传入 `(3, 32, 32)` 显示 RGB 图像。


## 字符语言模型示例

`language.txt` 是为 `examples/lm/language_model.ipynb` 编写的原创短中文语料，共 180 个字符（包括换行）。无需下载或外部数据包；词表按字符首次出现的顺序建立。示例只训练两步，用于验证整数编码、移位标签、因果网络、反向传播和逐字生成。
