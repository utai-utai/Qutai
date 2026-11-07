# 语言模型

1. [language_model.ipynb](language_model.ipynb)：字符编号、因果注意力、AdamW、裁剪、随机续写与旧教学模型的KV cache。
2. [decoder_training.ipynb](decoder_training.ipynb)：新的Dense/MoE Decoder，各20步真实训练、BF16激活/FP32状态、chunked loss与同一下一步的checkpoint恢复。固定整数toy数据，不下载模型或语料。
3. [train.py](train.py)：真实ByteLevel BPE语料的独立训练与显式恢复。接口细节见 [SDK LM说明](../../src/qutai/lm/README.md)。

配置放在 [configs](configs)：`lm_debug.json`为11,340,032参数，`dense300m.json`为306,226,176参数，`moe1b.json`为966,992,896总参数/306,390,016活跃参数。配置不是已训练模型，也不代表完成正式预训练。

在项目根目录运行：

```text
python examples/lm/train.py --config examples/lm/configs/lm_debug.json --tokenizer path/to/tokenizer.json --data path/to/pretrain --output runs/debug --device cuda --precision bfloat16 --batch-size 2 --sequence-length 128 --max-steps 1000 --checkpoint-every 100
```

新训练使用空的独立输出目录；恢复时重复原命令并添加 `--resume`。`--dry-run`只预检，不分配模型或写输出。旧 `examples/lm_train.py`仍可作为兼容入口，但推荐新路径。
