# 语言模型接口与训练入口

`lm.Tokenizer` 与 `lm.Model` 保留原有字符教学接口和 JSON 保存格式。下面的接口用于独立构建 Dense/MoE 训练流程，不改变旧 Notebook 的调用方式。

| 用途 | 接口 |
| --- | --- |
| 模型与参数预算 | `lm.Config`、`lm.Decoder`、`config.parameter_counts()` |
| 读取现有 ByteLevel BPE | `lm.ByteBPE(path)`，`encode` / `decode` / `token_to_id` / `sha256` |
| 小端整数语料 | `lm.TokenShards`、`lm.write_token_shards`、`lm.tokenize_shards` |
| 未 shift 的训练窗口与可恢复顺序 | `lm.PackedTokens`、`lm.StatefulBatcher` |
| 单次更新与普通训练循环 | `lm.train_step`、`lm.train`、`lm.cosine_learning_rate` |
| 二进制训练快照 | `lm.save_checkpoint`、`lm.load_checkpoint`、`lm.checkpoint_metadata` |
| 激活重计算 | `lm.activation_checkpoint(function, *args, parameters=...)` |
| BF16 计算、FP32 参数与状态 | `lm.autocast`，或训练函数的 `precision="bfloat16"` |

`ByteBPE` 读取无额外归一化和后处理的 ByteLevel BPE `tokenizer.json`。它按文件里的字节词表、合并优先级和 added tokens 编码，不自动插入 BOS/EOS；`decode` 默认跳过特殊 token，保留它们时传 `skip_special_tokens=False`。已有 UtaiLM 32768 词表可直接读取，SDK 不依赖 `tokenizers`。

语料 manifest 支持 `uint16-le` / `uint32-le`、`tokenizer_sha256` 和分片记录。`shards` 可以是记录列表，也可以是 `{"train": [...], "val": [...]}`；每条记录包含 `file`、`tokens`、`sha256`。`TokenShards` 使用只读 mmap，按需打开分片；`verify=True` 额外完整校验文件哈希。编码新语料时使用独立的新目录：

```python
from qutai import lm

tokenizer = lm.ByteBPE("path/to/tokenizer.json")
manifest = lm.tokenize_shards("new-corpus", texts, tokenizer,
                             bos_token_id=1, eos_token_id=2)
tokens = lm.TokenShards(manifest)
dataset = lm.PackedTokens(tokens, sequence_length=128)
batches = lm.StatefulBatcher(dataset, batch_size=2, seed=42, repeat=False)
inputs, labels = next(batches)
```

相邻窗口重叠一个边界 token，`inputs` 与 `labels` 都保持原始编号，`Decoder.loss` 内部只做一次 next-token shift。监督标签可用 `-100` 忽略位置。`StatefulBatcher.state_dict()` 保存 epoch、游标、排列和随机状态；默认不重复语料，需要多轮遍历时显式设置 `repeat=True`。

`train_step` 完成清梯度、若干微批前向与反向、范数检查/裁剪和一次优化器更新。不同微批的 LM 损失按有效监督 token 数加权，MoE 辅助损失按微批平均。`train` 只是逐步产生 metrics 的生成器，调用方可在每次完整更新后记录日志或保存快照。

二进制 checkpoint 保存模型配置、参数与共享关系、已有梯度和运行缓冲区、优化器参数组/计步/状态、Python RNG、显式传入的命名 RNG、数据游标、LR、token 数和训练计数。保存与恢复分块复制 CPU/CUDA 缓冲区，不把整份显存权重导出为 Python 列表。新分片全部落盘后才原子替换 manifest，失败时保留此前完整快照；恢复先验证全部分片哈希和结构，再原地写入现有参数。`checkpoint_metadata(path, verify=True)` 可在分配模型前验证快照。

```python
lm.save_checkpoint("runs/debug", model, optimizer, global_step=step,
                   tokens=tokens_seen, learning_rate=optimizer.learning_rate,
                   data=batches, tokenizer=tokenizer,
                   state={"accumulation_step": 0})
state = lm.load_checkpoint("runs/debug", model, optimizer,
                           data=batches, tokenizer=tokenizer)
```

在仓库根目录运行 `python -X utf8 examples/lm_train.py --help`。三个示例配置分别为 `examples/data/lm_debug.json`（11,340,032 参数）、`dense300m.json`（306,226,176 参数）和 `moe1b.json`（966,992,896 总参数，约 306,390,016 活跃参数）。后两者使用 UtaiLM 对应配置；配置文件描述结构，不代表已经完成正式预训练。

以下命令只向独立的 `runs/qutai-debug` 写模型与 JSONL 日志：

```text
python -X utf8 examples/lm_train.py --config examples/data/lm_debug.json --tokenizer path/to/tokenizer.json --data path/to/pretrain --output runs/qutai-debug --device cuda --precision bfloat16 --batch-size 2 --sequence-length 128 --max-steps 1000 --lr-total-steps 1000 --warmup-steps 50 --checkpoint-every 100
```

新训练要求输出目录为空。恢复时完整重复原命令并添加 `--resume`；模型、tokenizer 哈希、语料、批次、精度、总更新目标和 LR 预算必须相同。`--stop-after 100` 只限制本次执行的更新数，适合主动暂停，不会缩短保存的总预算。`--dry-run` 只预检数据/配置和参数预算，不分配模型或创建输出。

CLI 只在完整优化器更新边界保存，先清除已用完的梯度，快照只写参数和优化器状态。手工梯度累积需要保存中间梯度时，直接调用 `save_checkpoint`，不要先清梯度。

第一次 Ctrl+C 请求在当前完整更新结束后保存；再次 Ctrl+C 强制退出时，若当前更新未完成，就沿用此前完整 checkpoint。`metrics.jsonl` 每次更新写一条记录；恢复时以 checkpoint 为准，移除意外退出后尚未纳入快照的日志尾部。
