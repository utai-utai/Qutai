"""小而明确的 LM 训练循环；模型、优化器和 checkpoint 可以分别使用。"""

from math import cos, isfinite, pi
from sys import float_info

from ..nn.optimizers import clip_grad_norm
from .precision import autocast


def cosine_learning_rate(
    step: int,
    *,
    peak: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.1,
) -> float:
    """step 从零开始：线性预热，再按余弦降到 peak * min_ratio。"""
    if (
        step < 0
        or total_steps < 1
        or not 0 <= warmup_steps <= total_steps
        or not isfinite(peak)
        or peak < 0
        or not 0 <= min_ratio <= 1
    ):
        raise ValueError("学习率调度参数无效")
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if step >= total_steps:
        return peak * min_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * (min_ratio + (1 - min_ratio) * (1 + cos(pi * progress)) / 2)


def _token_counts(inputs, labels):
    labels = inputs if labels is None else labels
    if (
        not inputs
        or len(inputs) != len(labels)
        or any(
            len(row) < 2 or len(row) != len(target)
            for row, target in zip(inputs, labels)
        )
    ):
        raise ValueError("训练输入与标签需要相同的二维形状，序列至少有两个 token")
    total = sum(len(row) - 1 for row in inputs)
    supervised = sum(value != -100 for row in labels for value in row[1:])
    return total, supervised


def train_step(
    model,
    optimizer,
    batches,
    *,
    gradient_accumulation: int = 1,
    max_grad_norm: float | None = 1.0,
    chunk_size: int = 128,
    precision: str | None = None,
) -> dict:
    """完成一次优化器更新；标签保持未 shift，跨微批按有效 token 数加权。"""
    if gradient_accumulation < 1:
        raise ValueError("梯度累积次数必须为正数")
    batches = iter(batches)
    micro_batches = [next(batches) for _ in range(gradient_accumulation)]
    counts = [_token_counts(inputs, labels) for inputs, labels in micro_batches]
    supervised = sum(count for _, count in counts)
    tokens = sum(count for count, _ in counts)
    weights = [
        (count / supervised if supervised else total / tokens)
        for total, count in counts
    ]
    optimizer.zero_grad()
    losses, lm_losses, auxiliary_losses = [], [], []
    with autocast(precision is not None, dtype=precision or "bfloat16"):
        for (inputs, labels), weight in zip(micro_batches, weights):
            loss = model.loss(inputs, labels, train=True, chunk_size=chunk_size)
            if (
                getattr(model, "last_lm_loss", None) is not None
                and getattr(model, "last_auxiliary_loss", None) is not None
            ):
                lm_loss = model.last_lm_loss * weight
                auxiliary = model.last_auxiliary_loss / gradient_accumulation
                weighted = lm_loss + model.config.aux_loss_coef * auxiliary
                lm_losses.append(lm_loss)
                auxiliary_losses.append(auxiliary)
            else:
                weighted = loss * weight
            if weighted.requires_grad:
                weighted.backward(retain_graph=False)
            losses.append(weighted)
    values = [loss.item() for loss in losses]
    if any(not isfinite(value) for value in values):
        optimizer.zero_grad()
        raise ValueError("训练损失包含非有限值，本次没有更新参数")
    norm = clip_grad_norm(
        model, max_norm=max_grad_norm if max_grad_norm is not None else float_info.max
    )
    optimizer.step()
    metrics = {
        "loss": sum(values),
        "gradient_norm": norm,
        "tokens": tokens,
        "supervised_tokens": supervised,
        "micro_batches": gradient_accumulation,
        "learning_rate": optimizer.learning_rate,
    }
    if lm_losses:
        metrics.update(
            lm_loss=sum(loss.item() for loss in lm_losses),
            auxiliary_loss=sum(loss.item() for loss in auxiliary_losses),
        )
    return metrics


def train(
    model,
    optimizer,
    batches,
    *,
    steps: int,
    start_step: int = 0,
    tokens_seen: int = 0,
    schedule=None,
    gradient_accumulation: int = 1,
    max_grad_norm: float | None = 1.0,
    chunk_size: int = 128,
    precision: str | None = None,
):
    """逐步产生小的 metrics 字典；调用方可在 yield 后保存 checkpoint 或记录日志。"""
    if steps < 0 or start_step < 0 or tokens_seen < 0:
        raise ValueError("训练步数与 token 计数不能为负数")
    batches = iter(batches)
    for step in range(start_step, start_step + steps):
        if schedule is not None:
            optimizer.learning_rate = schedule(step)
        metrics = train_step(
            model,
            optimizer,
            batches,
            gradient_accumulation=gradient_accumulation,
            max_grad_norm=max_grad_norm,
            chunk_size=chunk_size,
            precision=precision,
        )
        tokens_seen += metrics["tokens"]
        yield {"global_step": step + 1, "tokens_seen": tokens_seen, **metrics}
