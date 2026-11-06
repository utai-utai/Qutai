"""语言模型专用 CPU 参考内核；CUDA 不调用这些计算。"""

from math import cos, exp, log, sin, sqrt
from random import Random
from . import cpu


def normal(size, mean=0.0, std=1.0, *, seed=None, dtype="float32"):
    random = Random(seed)
    return cpu.storage((random.gauss(mean, std) for _ in range(size)), dtype=dtype)


def gemm(
    a,
    b,
    rows,
    inner,
    cols,
    *,
    transpose_left=False,
    transpose_right=False,
    out_dtype=None,
    out=None,
    accumulate=False,
):
    if out_dtype is not None and out_dtype != a.dtype:
        raise ValueError("CPU GEMM 输出精度必须与输入相同")
    if accumulate and out is None:
        raise ValueError("累加 GEMM 需要提供输出存储")
    result = out if out is not None else cpu.full(rows * cols, dtype=a.dtype)
    if result.size != rows * cols or result.dtype != a.dtype:
        raise ValueError("GEMM 输出存储的大小或精度不匹配")
    if result is a or result is b:
        raise ValueError("GEMM 输出不能覆盖输入")
    for i in range(rows):
        for j in range(cols):
            value = 0.0
            for k in range(inner):
                value += (
                    a[k * rows + i if transpose_left else i * inner + k]
                    * b[j * inner + k if transpose_right else k * cols + j]
                )
            result[i * cols + j] = value + result[i * cols + j] if accumulate else value
    return result


def take(values, start, size):
    return values[start : start + size]


def add_into(target, source):
    for i in range(target.size):
        target[i] += source[i]


def embedding(weight, ids, width):
    return cpu.storage(
        (weight[token * width + j] for token in ids for j in range(width)),
        dtype=weight.dtype,
    )


def embedding_backward(gradient, ids, count, width):
    result = cpu.full(count * width, dtype=gradient.dtype)
    for i, token in enumerate(ids):
        for j in range(width):
            result[token * width + j] += gradient[i * width + j]
    return result


def rms_norm(values, weight, width, eps):
    out, inverse = (
        cpu.full(values.size, dtype=values.dtype),
        cpu.full(values.size // width, dtype=values.dtype),
    )
    for row in range(inverse.size):
        inv = 1 / sqrt(
            sum((values[row * width + j] ** 2 for j in range(width))) / width + eps
        )
        inverse[row] = inv
        for j in range(width):
            out[row * width + j] = values[row * width + j] * inv * weight[j]
    return (out, inverse)


def rms_norm_backward(gradient, values, weight, inverse, width):
    dx, dw = (
        cpu.full(values.size, dtype=values.dtype),
        cpu.full(width, dtype=values.dtype),
    )
    for row in range(inverse.size):
        inv = inverse[row]
        projected = (
            sum(
                (
                    gradient[row * width + j] * weight[j] * values[row * width + j]
                    for j in range(width)
                )
            )
            / width
        )
        for j in range(width):
            i = row * width + j
            dx[i] = inv * (gradient[i] * weight[j] - values[i] * inv * inv * projected)
            dw[j] += gradient[i] * values[i] * inv
    return (dx, dw)


def silu(values):
    return cpu.storage(
        (x * (1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))) for x in values),
        dtype=values.dtype,
    )


def silu_backward(gradient, values):

    def derivative(g, x):
        s = 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))
        return g * s * (1 + x * (1 - s))

    return cpu.storage(
        (derivative(g, x) for g, x in zip(gradient, values)), dtype=values.dtype
    )


def rope(values, batch, time, heads, dim, theta, offset=0, inverse=False):
    out = cpu.full(values.size, dtype=values.dtype)
    for b in range(batch):
        for t in range(time):
            for h in range(heads):
                for j in range(0, dim, 2):
                    angle = (offset + t) * theta ** (-j / dim) * (-1 if inverse else 1)
                    c, s = (cos(angle), sin(angle))
                    i = ((b * time + t) * heads + h) * dim + j
                    out[i], out[i + 1] = (
                        values[i] * c - values[i + 1] * s,
                        values[i] * s + values[i + 1] * c,
                    )
    return out


def attention(q, k, v, batch, time, heads, kv_heads, dim):
    out = cpu.full(q.size, dtype=q.dtype)
    lse = cpu.full(batch * time * heads, dtype=q.dtype)
    for b in range(batch):
        for t in range(time):
            for h in range(heads):
                qi = ((b * time + t) * heads + h) * dim
                kh = h // (heads // kv_heads)
                scores = [
                    sum(
                        (
                            q[qi + j] * k[((b * time + s) * kv_heads + kh) * dim + j]
                            for j in range(dim)
                        )
                    )
                    / sqrt(dim)
                    for s in range(t + 1)
                ]
                peak = max(scores)
                weights = [exp(x - peak) for x in scores]
                total = sum(weights)
                lse[(b * time + t) * heads + h] = peak + log(total)
                for j in range(dim):
                    out[qi + j] = (
                        sum(
                            (
                                p * v[((b * time + s) * kv_heads + kh) * dim + j]
                                for s, p in enumerate(weights)
                            )
                        )
                        / total
                    )
    return (out, lse)


def attention_backward(gradient, q, k, v, out, lse, batch, time, heads, kv_heads, dim):
    dq, dk, dv = (
        cpu.full(q.size, dtype=q.dtype),
        cpu.full(k.size, dtype=k.dtype),
        cpu.full(v.size, dtype=v.dtype),
    )
    scale = 1 / sqrt(dim)
    for b in range(batch):
        for t in range(time):
            for h in range(heads):
                qi = ((b * time + t) * heads + h) * dim
                kh = h // (heads // kv_heads)
                delta = sum((gradient[qi + j] * out[qi + j] for j in range(dim)))
                for s in range(t + 1):
                    ki = ((b * time + s) * kv_heads + kh) * dim
                    p = exp(
                        sum((q[qi + j] * k[ki + j] for j in range(dim))) * scale
                        - lse[(b * time + t) * heads + h]
                    )
                    dp = sum((gradient[qi + j] * v[ki + j] for j in range(dim)))
                    ds = p * (dp - delta) * scale
                    for j in range(dim):
                        dq[qi + j] += ds * k[ki + j]
                        dk[ki + j] += ds * q[qi + j]
                        dv[ki + j] += p * gradient[qi + j]
    return (dq, dk, dv)


def routing(scores, rows, experts, top_k):
    probs = cpu.softmax(scores, (rows, experts), 1)
    weights = cpu.full(rows * top_k, dtype=scores.dtype)
    indices = cpu.full(rows * top_k, dtype="int64")
    token_ids = [[] for _ in range(experts)]
    slot_ids = [[] for _ in range(experts)]
    for row in range(rows):
        selected = sorted(
            range(experts), key=lambda e: probs[row * experts + e], reverse=True
        )[:top_k]
        total = sum((probs[row * experts + e] for e in selected))
        for slot, e in enumerate(selected):
            indices[row * top_k + slot] = e
            weights[row * top_k + slot] = probs[row * experts + e] / total
            token_ids[e].append(row)
            slot_ids[e].append(slot)
    counts = tuple(map(len, token_ids))
    utilization = cpu.storage((c / (rows * top_k) for c in counts), dtype=scores.dtype)
    aux = experts * sum(
        (
            utilization[e]
            * sum((probs[row * experts + e] for row in range(rows)))
            / rows
            for e in range(experts)
        )
    )
    return dict(
        probs=probs,
        weights=weights,
        indices=indices,
        counts=counts,
        ids=token_ids,
        slots=slot_ids,
        utilization=utilization,
        aux=cpu.storage([aux], dtype=scores.dtype),
        rows=rows,
        experts=experts,
        top_k=top_k,
    )


def routing_backward(gradient, route, auxiliary=False):
    rows, experts, top_k = (route[n] for n in ("rows", "experts", "top_k"))
    out = cpu.full(rows * experts, dtype=gradient.dtype)
    for row in range(rows):
        if auxiliary:
            center = sum(
                (
                    route["probs"][row * experts + e] * route["utilization"][e]
                    for e in range(experts)
                )
            )
            for e in range(experts):
                out[row * experts + e] = (
                    gradient[0]
                    * experts
                    / rows
                    * route["probs"][row * experts + e]
                    * (route["utilization"][e] - center)
                )
        else:
            center = sum(
                (
                    gradient[row * top_k + s] * route["weights"][row * top_k + s]
                    for s in range(top_k)
                )
            )
            for s in range(top_k):
                e = route["indices"][row * top_k + s]
                out[row * experts + e] = route["weights"][row * top_k + s] * (
                    gradient[row * top_k + s] - center
                )
    return out


def route_gather(values, route, expert, width):
    return cpu.storage(
        (values[row * width + j] for row in route["ids"][expert] for j in range(width)),
        dtype=values.dtype,
    )


def route_gather_backward(gradient, route, expert, width):
    out = cpu.full(route["rows"] * width, dtype=gradient.dtype)
    for i, row in enumerate(route["ids"][expert]):
        for j in range(width):
            out[row * width + j] += gradient[i * width + j]
    return out


def combine(outputs, weights, route, width):
    result = cpu.full(route["rows"] * width, dtype=weights.dtype)
    for expert, values in outputs:
        for i, (row, slot) in enumerate(
            zip(route["ids"][expert], route["slots"][expert])
        ):
            factor = weights[row * route["top_k"] + slot]
            for j in range(width):
                result[row * width + j] += factor * values[i * width + j]
    return result


def combine_backward(gradient, outputs, weights, route, width):
    grads = []
    dw = cpu.full(weights.size, dtype=weights.dtype)
    for expert, values in outputs:
        dy = cpu.full(values.size, dtype=values.dtype)
        for i, (row, slot) in enumerate(
            zip(route["ids"][expert], route["slots"][expert])
        ):
            at = row * route["top_k"] + slot
            factor = weights[at]
            for j in range(width):
                dy[i * width + j] = gradient[row * width + j] * factor
                dw[at] += gradient[row * width + j] * values[i * width + j]
        grads.append(dy)
    return (*grads, dw)


def head_rows(values, start, count, width, time):
    return cpu.storage(
        (
            values[(row // (time - 1) * time + row % (time - 1)) * width + j]
            for row in range(start, start + count)
            for j in range(width)
        ),
        dtype=values.dtype,
    )


def head_rows_backward(target, gradient, start, count, width, time):
    for row in range(count):
        at = (start + row) // (time - 1) * time + (start + row) % (time - 1)
        for j in range(width):
            target[at * width + j] += gradient[row * width + j]


def head_loss(logits, labels, rows, vocab):
    result = 0.0
    for row, label in enumerate(labels):
        if label == -100:
            continue
        peak = max((logits[row * vocab + j] for j in range(vocab)))
        result += log(
            sum((exp(logits[row * vocab + j] - peak) for j in range(vocab)))
        ) - (logits[row * vocab + label] - peak)
    return cpu.storage([result], dtype=logits.dtype)


def head_gradient(logits, labels, rows, vocab, gradient, count):
    for row, label in enumerate(labels):
        peak = max((logits[row * vocab + j] for j in range(vocab)))
        total = sum((exp(logits[row * vocab + j] - peak) for j in range(vocab)))
        for j in range(vocab):
            logits[row * vocab + j] = (
                0
                if label == -100
                else gradient[0]
                / count
                * (exp(logits[row * vocab + j] - peak) / total - (j == label))
            )
    return logits
