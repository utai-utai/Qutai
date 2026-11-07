import argparse
import json
import math
import random
import signal
from dataclasses import asdict
from pathlib import Path

from qutai import lm, nn
from qutai.lm.bpe import SPECIAL_TOKENS


def arguments():
    parser = argparse.ArgumentParser(description="LM 训练与恢复")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="带 tokenizer_sha256 的语料 manifest 或目录",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="本次训练独立输出目录"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="完整训练的总更新目标，恢复时保持一致",
    )
    parser.add_argument(
        "--lr-total-steps", type=int, help="完整 LR 调度预算，默认等于 max-steps"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="允许语料用尽后重复")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--verify-data", action="store_true", help="启动时额外完整核对各 token 分片哈希"
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--stop-after", type=int, help="本次最多执行多少次完整更新，不改变总训练/LR预算"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只做配置/数据预检，不分配模型或写输出"
    )
    args = parser.parse_args()
    return parser, args


def run_configuration(args, config, tokenizer, dataset):
    return {
        "model": asdict(config),
        "tokenizer_sha256": tokenizer.sha256,
        "data_signature": dataset.signature,
        "split": args.split,
        "device": args.device,
        "dtype": args.dtype,
        "precision": args.precision,
        "max_steps": args.max_steps,
        "lr_total_steps": args.lr_total_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "accumulation_steps": args.accumulation_steps,
        "loss_chunk_size": args.loss_chunk_size,
        "betas": [args.beta1, args.beta2],
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "repeat": args.repeat,
    }


def validate_output(args):
    output = args.output.resolve()
    data = args.data.resolve()
    data = data.parent if data.is_file() else data
    protected = (args.config.resolve(), args.tokenizer.resolve())
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("训练输出目录必须与输入语料目录分开")
    if any(path == output or path.is_relative_to(output) for path in protected):
        raise ValueError("训练输出目录不能包含输入配置或 tokenizer")
    manifest = output / "manifest.json"
    if args.resume:
        if not manifest.is_file():
            raise ValueError("--resume 需要输出目录中已有完整 checkpoint")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("新训练需要空的独立输出目录；接续训练请显式使用 --resume")


def truncate_metrics(path, step):
    """checkpoint 是恢复依据，去掉意外退出后尚未纳入 checkpoint 的日志尾部。"""
    if not path.exists():
        return
    temporary = path.with_suffix(".tmp")
    with (
        path.open(encoding="utf8") as source,
        temporary.open("w", encoding="utf8") as target,
    ):
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record["global_step"] > step:
                break
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_resume(saved, configuration, batcher):
    state = saved["training"]
    if (
        state["extra"].get("run_config") != configuration
        or saved["model_config"] != configuration["model"]
        or saved["tokenizer_sha256"] != configuration["tokenizer_sha256"]
    ):
        raise ValueError(
            "恢复配置与 checkpoint 不一致；不能改变模型、语料、批次或训练/LR预算"
        )
    expected_data = {
        name: configuration[name]
        for name in ("seed", "shuffle", "repeat", "batch_size")
    }
    if any(state["data"][name] != value for name, value in expected_data.items()):
        raise ValueError("checkpoint 的 batcher 配置与训练记录不一致")
    batcher.load_state_dict(state["data"])
    step = state["global_step"]
    samples = step * configuration["batch_size"] * configuration["accumulation_steps"]
    if (
        not 0 <= step <= configuration["max_steps"]
        or state["extra"].get("accumulation_step") != 0
        or batcher.epoch * len(batcher.dataset) + batcher.cursor != samples
        or state["tokens"] != samples * (configuration["sequence_length"] - 1)
    ):
        raise ValueError("checkpoint 的更新边界、数据游标或 token 计数不一致")
    rate = (
        configuration["learning_rate"]
        if step == 0
        else lm.cosine_learning_rate(
            step - 1,
            peak=configuration["learning_rate"],
            total_steps=configuration["lr_total_steps"],
            warmup_steps=configuration["warmup_steps"],
            min_ratio=configuration["min_lr_ratio"],
        )
    )
    expected_settings = {
        "learning_rate": rate,
        "betas": configuration["betas"],
        "eps": configuration["eps"],
        "weight_decay": configuration["weight_decay"],
    }
    groups = [
        {
            "parameters": [
                record["tensor"]
                for record in saved["parameters"]
                if len(record["shape"]) >= 2
            ],
            "settings": {"weight_decay": configuration["weight_decay"]},
        },
        {
            "parameters": [
                record["tensor"]
                for record in saved["parameters"]
                if len(record["shape"]) < 2
            ],
            "settings": {"weight_decay": 0.0},
        },
    ]
    optimizer = saved["optimizer"]
    if (
        optimizer is None
        or optimizer["type"] != "AdamW"
        or optimizer["settings"] != expected_settings
        or optimizer.get("groups") != groups
        or state["learning_rate"] != rate
        or any(
            type(count) is not int or not 1 <= count <= step
            for count in optimizer["steps"].values()
        )
    ):
        raise ValueError("checkpoint 的 AdamW 参数组、计步或 LR 状态与训练记录不一致")


def main():
    parser, args = arguments()
    try:
        config = lm.Config(**json.loads(args.config.read_text(encoding="utf8")))
        tokenizer = lm.ByteBPE(args.tokenizer)
        args.lr_total_steps = (
            args.max_steps if args.lr_total_steps is None else args.lr_total_steps
        )
        args.sequence_length = (
            config.max_seq_len if args.sequence_length is None else args.sequence_length
        )
        if config.vocab_size != tokenizer.size:
            raise ValueError("模型词表大小与 tokenizer 不一致")
        if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != list(
            range(10)
        ):
            raise ValueError("UtaiLM 特殊 token 必须保持固定的 0 到 9 编号")
        if not 2 <= args.sequence_length <= config.max_seq_len:
            raise ValueError("sequence-length 必须在 2 到模型上下文长度之间")
        if (
            min(
                args.max_steps,
                args.batch_size,
                args.accumulation_steps,
                args.loss_chunk_size,
                args.checkpoint_every,
                args.log_every,
            )
            < 1
        ):
            raise ValueError("训练步数、批次、累积和记录间隔必须为正数")
        if args.stop_after is not None and args.stop_after < 1:
            raise ValueError("stop-after 必须为正数")
        if (
            args.lr_total_steps < args.max_steps
            or not 0 <= args.warmup_steps <= args.lr_total_steps
        ):
            raise ValueError("LR 调度预算必须覆盖 max-steps 与 warmup-steps")
        if not all(
            math.isfinite(value)
            for value in (
                args.learning_rate,
                args.min_lr_ratio,
                args.beta1,
                args.beta2,
                args.eps,
                args.weight_decay,
                args.max_grad_norm,
            )
        ):
            raise ValueError("训练超参数必须是有限数")
        if (
            args.learning_rate < 0
            or not 0 <= args.min_lr_ratio <= 1
            or args.eps < 0
            or args.weight_decay < 0
            or args.max_grad_norm < 0
            or not 0 <= args.beta1 < 1
            or not 0 <= args.beta2 < 1
        ):
            raise ValueError("训练超参数超出有效范围")
        if args.precision == "bfloat16" and (
            args.device != "cuda" or args.dtype != "float32"
        ):
            raise ValueError("bfloat16 计算需要 CUDA 与 float32 主参数")
        validate_output(args)
        with lm.TokenShards(
            args.data, split=args.split, verify=args.verify_data
        ) as tokens:
            if tokens.tokenizer_hash != tokenizer.sha256:
                raise ValueError("语料的 tokenizer_sha256 与当前 tokenizer 不一致")
            dataset = lm.PackedTokens(tokens, args.sequence_length)
            batcher = lm.StatefulBatcher(
                dataset,
                batch_size=args.batch_size,
                seed=args.seed,
                shuffle=not args.no_shuffle,
                repeat=args.repeat,
            )
            configuration = run_configuration(args, config, tokenizer, dataset)
            saved = (
                lm.checkpoint_metadata(args.output, verify=True)
                if args.resume
                else None
            )
            if saved is not None:
                validate_resume(saved, configuration, batcher)
            if (
                not args.repeat
                and len(dataset)
                < args.max_steps * args.batch_size * args.accumulation_steps
            ):
                raise ValueError("语料不足以覆盖完整更新预算；减少预算或显式 --repeat")
            if saved is None or saved["training"]["global_step"] < args.max_steps:
                cursor = batcher.state_dict()
                inputs, labels = next(batcher)
                batcher.load_state_dict(cursor)
                if any(
                    not 0 <= token < config.vocab_size
                    for row in inputs
                    for token in row
                ):
                    raise ValueError("下一批语料 token 超出模型词表范围")
            print(
                json.dumps(
                    {
                        "preflight": "passed",
                        "parameter_counts": config.parameter_counts(),
                        "sequences": len(dataset),
                        "run_config": configuration,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.dry_run:
                return
            run(args, config, tokenizer, batcher, configuration)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


def run(args, config, tokenizer, batcher, configuration):
    random.seed(args.seed)
    model = lm.Decoder(config, device=args.device, dtype=args.dtype, seed=args.seed)
    decay = tuple(parameter for parameter in model.parameters() if parameter.ndim >= 2)
    no_decay = tuple(
        parameter for parameter in model.parameters() if parameter.ndim < 2
    )
    optimizer = nn.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    step, tokens_seen = 0, 0
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    if args.resume:
        restored = lm.load_checkpoint(
            args.output, model, optimizer, data=batcher, tokenizer=tokenizer
        )
        step, tokens_seen = restored["global_step"], restored["tokens"]
        truncate_metrics(metrics_path, step)

    def save():
        # 只在初始化或完整更新边界调用；下一步也会清梯度，无需把旧梯度写入正式快照。
        optimizer.zero_grad()
        return lm.save_checkpoint(
            args.output,
            model,
            optimizer,
            global_step=step,
            tokens=tokens_seen,
            learning_rate=optimizer.learning_rate,
            data=batcher,
            tokenizer=tokenizer,
            state={"run_config": configuration, "accumulation_step": 0},
        )

    if not args.resume:
        save()
    stop_requested, boundary = False, True

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True

    old_handler = signal.signal(signal.SIGINT, request_stop)
    completed = 0
    try:
        with metrics_path.open("a", encoding="utf8") as metrics_file:
            while step < args.max_steps and (
                args.stop_after is None or completed < args.stop_after
            ):
                if stop_requested:
                    break
                boundary = False
                optimizer.learning_rate = lm.cosine_learning_rate(
                    step,
                    peak=args.learning_rate,
                    total_steps=args.lr_total_steps,
                    warmup_steps=args.warmup_steps,
                    min_ratio=args.min_lr_ratio,
                )
                metrics = lm.train_step(
                    model,
                    optimizer,
                    batcher,
                    gradient_accumulation=args.accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                    chunk_size=args.loss_chunk_size,
                    precision="bfloat16" if args.precision == "bfloat16" else None,
                )
                step += 1
                tokens_seen += metrics["tokens"]
                completed += 1
                boundary = True
                record = {"global_step": step, "tokens_seen": tokens_seen, **metrics}
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                metrics_file.write(line + "\n")
                metrics_file.flush()
                if step % args.log_every == 0 or step == args.max_steps:
                    print(line, flush=True)
                if step % args.checkpoint_every == 0:
                    save()
        save()
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output / "manifest.json"),
                    "global_step": step,
                    "tokens": tokens_seen,
                    "reason": "complete" if step == args.max_steps else "paused",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except KeyboardInterrupt:
        if boundary:
            save()
            print(f"已在完整更新边界保存：step {step}", flush=True)
        else:
            print("当前更新未完成，保留此前完整 checkpoint。", flush=True)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    main()
