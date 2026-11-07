"""语言模型 CUDA 调度；路由索引和数值计算均留在显存。"""

from array import array
from pathlib import Path
from . import cuda
from .cuda_runtime import get_runtime

KERNEL = Path(__file__).parent / "kernels/lm.cu"


def _acc_dtype(dtype):
    return "float64" if dtype == "float64" else "float32"


def _run(name, size, *arguments, dtype="float32", block=False, block_size=256):
    get_runtime().launch(
        KERNEL,
        name,
        size * block_size if block else size,
        list(arguments),
        options=(
            "-DREAL=" + ("double" if dtype == "float64" else "float"),
            "-DACT_BF16=" + ("1" if dtype == "bfloat16" else "0"),
        ),
        block_size=block_size,
    )


def normal(size, mean=0.0, std=1.0, *, seed=None, dtype="float32"):
    return cuda.normal(size, mean=mean, std=std, seed=seed, dtype=dtype)


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
    return cuda.gemm(
        a,
        b,
        rows,
        inner,
        cols,
        transpose_left=transpose_left,
        transpose_right=transpose_right,
        out_dtype=out_dtype,
        out=out,
        accumulate=accumulate,
    )


def take(values, start, size):
    result = cuda.CUDAStorage(size, values.dtype)
    get_runtime().copy(
        result.ptr, values.ptr + start * cuda._TYPES[values.dtype][1], result.nbytes
    )
    return result


def add_into(target, source):
    _run(
        "lm_add_into",
        target.size,
        cuda._p(target),
        cuda._p(source),
        cuda._i(target.size),
        dtype=target.dtype,
    )


def embedding(weight, ids, width, *, out_dtype=None):
    dtype = out_dtype or weight.dtype
    if _acc_dtype(dtype) != weight.dtype:
        raise ValueError("嵌入参数精度与激活累积精度不一致")
    identifiers = cuda.storage(ids, dtype="int64")
    result = cuda.CUDAStorage(len(ids) * width, dtype)
    _run(
        "lm_embedding",
        result.size,
        cuda._p(weight),
        cuda._p(identifiers),
        cuda._p(result),
        cuda._i(result.size),
        cuda._i(width),
        dtype=dtype,
    )
    return result


def embedding_backward(gradient, ids, count, width):
    groups = {}
    for row, token in enumerate(ids):
        groups.setdefault(token, []).append(row)
    tokens = sorted(groups)
    offsets = [0]
    positions = []
    for token in tokens:
        positions.extend(groups[token])
        offsets.append(len(positions))
    identifiers = cuda.storage(tokens, dtype="int64")
    boundaries = cuda.storage(offsets, dtype="int64")
    rows = cuda.storage(positions, dtype="int64")
    result = cuda.full(count * width, dtype=_acc_dtype(gradient.dtype))
    size = len(tokens) * width
    _run(
        "lm_embedding_backward",
        size,
        cuda._p(gradient),
        cuda._p(identifiers),
        cuda._p(boundaries),
        cuda._p(rows),
        cuda._p(result),
        cuda._i(size),
        cuda._i(width),
        dtype=gradient.dtype,
    )
    return result


def rms_norm(values, weight, width, eps):
    rows = values.size // width
    out, inverse = (
        cuda.CUDAStorage(values.size, _acc_dtype(values.dtype)),
        cuda.CUDAStorage(rows, _acc_dtype(values.dtype)),
    )
    _run(
        "lm_rms",
        rows,
        cuda._p(values),
        cuda._p(weight),
        cuda._p(out),
        cuda._p(inverse),
        cuda._i(rows),
        cuda._i(width),
        cuda._s(eps, _acc_dtype(values.dtype)),
        dtype=values.dtype,
        block=True,
    )
    return (out, inverse)


def rms_norm_backward(gradient, values, weight, inverse, width):
    dx, dw = (
        cuda.CUDAStorage(values.size, values.dtype),
        cuda.CUDAStorage(width, weight.dtype),
    )
    _run(
        "lm_rms_dx",
        inverse.size,
        cuda._p(gradient),
        cuda._p(values),
        cuda._p(weight),
        cuda._p(inverse),
        cuda._p(dx),
        cuda._i(inverse.size),
        cuda._i(width),
        dtype=values.dtype,
        block=True,
    )
    _run(
        "lm_rms_dw",
        width,
        cuda._p(gradient),
        cuda._p(values),
        cuda._p(inverse),
        cuda._p(dw),
        cuda._i(width),
        cuda._i(inverse.size),
        dtype=values.dtype,
    )
    return (dx, dw)


def silu(values):
    out = cuda.CUDAStorage(values.size, values.dtype)
    _run(
        "lm_silu",
        values.size,
        cuda._p(values),
        cuda._p(None),
        cuda._p(out),
        cuda._i(values.size),
        cuda._i(0),
        dtype=values.dtype,
    )
    return out


def silu_backward(gradient, values):
    out = cuda.CUDAStorage(values.size, values.dtype)
    _run(
        "lm_silu",
        values.size,
        cuda._p(values),
        cuda._p(gradient),
        cuda._p(out),
        cuda._i(values.size),
        cuda._i(1),
        dtype=values.dtype,
    )
    return out


def rope(values, batch, time, heads, dim, theta, offset=0, inverse=False):
    result = cuda.CUDAStorage(values.size, values.dtype)
    _run(
        "lm_rope",
        values.size // 2,
        cuda._p(values),
        cuda._p(result),
        cuda._i(values.size // 2),
        cuda._i(time),
        cuda._i(heads),
        cuda._i(dim),
        cuda._s(theta, _acc_dtype(values.dtype)),
        cuda._i(offset),
        cuda._i(inverse),
        dtype=values.dtype,
    )
    return result


def attention(q, k, v, batch, time, heads, kv_heads, dim):
    if q.dtype == k.dtype == v.dtype == "bfloat16" and dim == 64 and time >= 16:
        out = cuda.CUDAStorage(q.size, q.dtype)
        lse = cuda.CUDAStorage(batch * time * heads, "float32")
        get_runtime().launch(
            KERNEL,
            "lm_attention_bf16_d64",
            q.size,
            [
                cuda._p(q),
                cuda._p(k),
                cuda._p(v),
                cuda._p(out),
                cuda._p(lse),
                cuda._i(time),
                cuda._i(heads),
                cuda._i(kv_heads),
            ],
            options=("-DREAL=float", "-DACT_BF16=1"),
            grid=((time + 15) // 16, heads, batch),
            block=(128, 1, 1),
        )
        return out, lse
    return _attention_warp(q, k, v, batch, time, heads, kv_heads, dim)


def _attention_warp(q, k, v, batch, time, heads, kv_heads, dim):
    if len({q.dtype, k.dtype, v.dtype}) != 1:
        raise ValueError("Attention 的 Q/K/V 精度必须一致")
    out, lse = (
        cuda.CUDAStorage(q.size, q.dtype),
        cuda.CUDAStorage(batch * time * heads, _acc_dtype(q.dtype)),
    )
    _run(
        "lm_attention",
        lse.size,
        cuda._p(q),
        cuda._p(k),
        cuda._p(v),
        cuda._p(out),
        cuda._p(lse),
        *(cuda._i(x) for x in (batch, time, heads, kv_heads, dim)),
        dtype=q.dtype,
        block=True,
        block_size=32,
    )
    return (out, lse)


def attention_backward(gradient, q, k, v, out, lse, batch, time, heads, kv_heads, dim):
    if len(
        {gradient.dtype, q.dtype, k.dtype, v.dtype, out.dtype}
    ) != 1 or lse.dtype != _acc_dtype(q.dtype):
        raise ValueError("Attention 反向的激活和梯度精度必须一致，LSE 使用累加精度")
    dq = cuda.CUDAStorage(q.size, q.dtype)
    dk = cuda.CUDAStorage(k.size, k.dtype)
    dv = cuda.CUDAStorage(v.size, v.dtype)
    delta = cuda.CUDAStorage(lse.size, _acc_dtype(q.dtype))
    dims = tuple(cuda._i(x) for x in (batch, time, heads, kv_heads, dim))
    _run(
        "lm_attention_delta",
        lse.size,
        cuda._p(gradient),
        cuda._p(out),
        cuda._p(delta),
        cuda._i(lse.size),
        cuda._i(dim),
        dtype=q.dtype,
        block=True,
        block_size=32,
    )
    if (
        q.dtype == k.dtype == v.dtype == gradient.dtype == "bfloat16"
        and dim == 64
        and time >= 64
    ):
        arguments = [
            cuda._p(gradient),
            cuda._p(q),
            cuda._p(k),
            cuda._p(v),
            cuda._p(lse),
            cuda._p(delta),
        ]
        dimensions = [cuda._i(time), cuda._i(heads), cuda._i(kv_heads)]
        runtime = get_runtime()
        runtime.launch(
            KERNEL,
            "lm_attention_bf16_dq",
            q.size,
            arguments + [cuda._p(dq)] + dimensions,
            options=("-DREAL=float", "-DACT_BF16=1"),
            grid=((time + 15) // 16, heads, batch),
            block=(128, 1, 1),
        )
        runtime.launch(
            KERNEL,
            "lm_attention_bf16_dkv",
            k.size,
            arguments + [cuda._p(dk), cuda._p(dv)] + dimensions,
            options=("-DREAL=float", "-DACT_BF16=1"),
            grid=((time + 15) // 16, kv_heads, batch),
            block=(128, 1, 1),
        )
        return (dq, dk, dv)
    _run(
        "lm_attention_backward_q",
        lse.size,
        cuda._p(gradient),
        cuda._p(q),
        cuda._p(k),
        cuda._p(v),
        cuda._p(lse),
        cuda._p(delta),
        cuda._p(dq),
        *dims,
        dtype=q.dtype,
        block=True,
        block_size=32,
    )
    _run(
        "lm_attention_backward_kv",
        batch * time * kv_heads,
        cuda._p(gradient),
        cuda._p(q),
        cuda._p(k),
        cuda._p(v),
        cuda._p(lse),
        cuda._p(delta),
        cuda._p(dk),
        cuda._p(dv),
        *dims,
        dtype=q.dtype,
        block=True,
        block_size=32,
    )
    return (dq, dk, dv)


def routing(scores, rows, experts, top_k):
    if scores.dtype == "bfloat16":
        scores = cuda.from_storage(scores, dtype="float32")
    probs = cuda.softmax(scores, (rows, experts), 1)
    capacity = rows * top_k
    weights, indices = (
        cuda.CUDAStorage(capacity, scores.dtype),
        cuda.CUDAStorage(capacity, "int64"),
    )
    count_buffer = cuda.full(experts, dtype="int64")
    ids, slots = (
        cuda.CUDAStorage(experts * capacity, "int64"),
        cuda.CUDAStorage(experts * capacity, "int64"),
    )
    locations = cuda.CUDAStorage(capacity, "int64")
    _run(
        "lm_route",
        rows,
        cuda._p(probs),
        cuda._p(weights),
        cuda._p(indices),
        cuda._i(rows),
        cuda._i(experts),
        cuda._i(top_k),
        dtype=scores.dtype,
    )
    _run(
        "lm_route_pack",
        experts,
        cuda._p(indices),
        cuda._p(count_buffer),
        cuda._p(ids),
        cuda._p(slots),
        cuda._p(locations),
        cuda._i(capacity),
        cuda._i(capacity),
        cuda._i(top_k),
        cuda._i(experts),
        dtype=scores.dtype,
        block=True,
    )
    utilization, aux = (
        cuda.CUDAStorage(experts, scores.dtype),
        cuda.CUDAStorage(1, scores.dtype),
    )
    _run(
        "lm_route_util",
        experts,
        cuda._p(count_buffer),
        cuda._p(utilization),
        cuda._i(experts),
        cuda._i(capacity),
        dtype=scores.dtype,
    )
    _run(
        "lm_route_aux",
        1,
        cuda._p(probs),
        cuda._p(utilization),
        cuda._p(aux),
        cuda._i(rows),
        cuda._i(experts),
        dtype=scores.dtype,
    )
    # 整数计数仅决定 GEMM 形状；token/slot 路由索引始终留在显存。
    counts = array("q", [0]) * experts
    get_runtime().download(count_buffer.ptr, counts)
    return dict(
        probs=probs,
        weights=weights,
        indices=indices,
        counts=tuple(counts),
        ids=ids,
        slots=slots,
        locations=locations,
        utilization=utilization,
        aux=aux,
        rows=rows,
        experts=experts,
        top_k=top_k,
        capacity=capacity,
    )


def routing_backward(gradient, route, auxiliary=False):
    result = cuda.CUDAStorage(route["rows"] * route["experts"], gradient.dtype)
    _run(
        "lm_route_backward",
        result.size,
        cuda._p(gradient),
        cuda._p(route["probs"]),
        cuda._p(route["weights"]),
        cuda._p(route["indices"]),
        cuda._p(route["utilization"]),
        cuda._p(result),
        cuda._i(result.size),
        *(cuda._i(route[k]) for k in ("rows", "experts", "top_k")),
        cuda._i(auxiliary),
        dtype=gradient.dtype,
    )
    return result


def route_gather(values, route, expert, width):
    result = cuda.CUDAStorage(route["counts"][expert] * width, values.dtype)
    _run(
        "lm_route_gather",
        result.size,
        cuda._p(values),
        cuda._p(route["ids"]),
        cuda._p(result),
        cuda._i(result.size),
        cuda._i(width),
        cuda._i(expert * route["capacity"]),
        cuda._i(0),
        dtype=values.dtype,
    )
    return result


def route_gather_backward(gradient, route, expert, width):
    result = cuda.full(route["rows"] * width, dtype=gradient.dtype)
    _run(
        "lm_route_gather",
        gradient.size,
        cuda._p(gradient),
        cuda._p(route["ids"]),
        cuda._p(result),
        cuda._i(gradient.size),
        cuda._i(width),
        cuda._i(expert * route["capacity"]),
        cuda._i(1),
        dtype=gradient.dtype,
    )
    return result


def _pointer_table(values):
    """小型专家地址表经 kernel 参数写入，避免每层同步 HtoD 复制。"""
    result = cuda.CUDAStorage(len(values), "int64")
    for start in range(0, len(values), 8):
        part = values[start : start + 8]
        _run(
            "lm_pointer_table",
            len(part),
            cuda._p(result),
            cuda._i(start),
            cuda._i(len(part)),
            *(cuda._i(value) for value in (*part, *([0] * (8 - len(part))))),
        )
    return result


def combine(outputs, weights, route, width):
    dtype = outputs[0][1].dtype if outputs else weights.dtype
    result = cuda.CUDAStorage(route["rows"] * width, dtype)
    pointers = [0] * route["experts"]
    for expert, values in outputs:
        pointers[expert] = values.ptr
    addresses = _pointer_table(pointers)
    _run(
        "lm_combine_all",
        result.size,
        cuda._p(addresses),
        cuda._p(weights),
        cuda._p(route["indices"]),
        cuda._p(route["locations"]),
        cuda._p(result),
        cuda._i(result.size),
        cuda._i(width),
        cuda._i(route["top_k"]),
        dtype=dtype,
    )
    return result


def combine_backward(gradient, outputs, weights, route, width):
    dw = cuda.full(weights.size, dtype=weights.dtype)
    grads = []
    for expert, values in outputs:
        dy = cuda.CUDAStorage(values.size, values.dtype)
        _run(
            "lm_combine_backward",
            values.size,
            cuda._p(gradient),
            cuda._p(values),
            cuda._p(weights),
            cuda._p(route["ids"]),
            cuda._p(route["slots"]),
            cuda._p(dy),
            cuda._p(dw),
            cuda._i(values.size),
            cuda._i(width),
            cuda._i(expert * route["capacity"]),
            cuda._i(route["top_k"]),
            dtype=values.dtype,
        )
        _run(
            "lm_combine_weight_backward",
            route["counts"][expert],
            cuda._p(gradient),
            cuda._p(values),
            cuda._p(route["ids"]),
            cuda._p(route["slots"]),
            cuda._p(dw),
            cuda._i(route["counts"][expert]),
            cuda._i(width),
            cuda._i(expert * route["capacity"]),
            cuda._i(route["top_k"]),
            dtype=values.dtype,
            block=True,
        )
        grads.append(dy)
    return (*grads, dw)


def head_rows(values, start, count, width, time):
    result = cuda.CUDAStorage(count * width, values.dtype)
    _run(
        "lm_head_rows",
        result.size,
        cuda._p(values),
        cuda._p(result),
        cuda._i(result.size),
        cuda._i(start),
        cuda._i(width),
        cuda._i(time),
        cuda._i(0),
        dtype=values.dtype,
    )
    return result


def head_rows_backward(target, gradient, start, count, width, time):
    _run(
        "lm_head_rows",
        gradient.size,
        cuda._p(gradient),
        cuda._p(target),
        cuda._i(gradient.size),
        cuda._i(start),
        cuda._i(width),
        cuda._i(time),
        cuda._i(1),
        dtype=target.dtype,
    )


def prepare_labels(labels):
    return cuda.storage(labels, dtype="int64")


def _head_labels(labels, start, rows):
    ids = labels if isinstance(labels, cuda.CUDAStorage) else prepare_labels(labels)
    if ids.dtype != "int64" or start < 0 or start + rows > ids.size:
        raise ValueError("输出头标签存储或范围无效")
    pointer = cuda._p(ids)
    pointer.value += start * 8
    return ids, pointer


def head_loss(logits, labels, rows, vocab, *, start=0):
    ids, pointer = _head_labels(labels, start, rows)
    values = cuda.CUDAStorage(rows, _acc_dtype(logits.dtype))
    _run(
        "lm_head_loss",
        rows,
        cuda._p(logits),
        pointer,
        cuda._p(values),
        cuda._i(rows),
        cuda._i(vocab),
        dtype=logits.dtype,
        block=True,
    )
    return cuda.sum(values)


def head_gradient(logits, labels, rows, vocab, gradient, count, *, start=0):
    ids, pointer = _head_labels(labels, start, rows)
    _run(
        "lm_head_gradient",
        rows,
        cuda._p(logits),
        pointer,
        cuda._p(gradient),
        cuda._i(rows),
        cuda._i(vocab),
        cuda._s(1 / count, _acc_dtype(logits.dtype)),
        dtype=logits.dtype,
        block=True,
    )
    return logits
