"""Decoder 的 Apple 实现：复用 Metal 数学运算，路由只读取整数批次大小。"""

from math import sqrt
from random import Random

from . import apple


def normal(size, mean=0.0, std=1.0, *, seed=None, dtype="float32"):
    random = Random(seed)
    return apple.storage((random.gauss(mean, std) for _ in range(size)), dtype=dtype)


def gemm(a, b, rows, inner, cols, *, transpose_left=False, transpose_right=False, out_dtype=None, out=None, accumulate=False):
    if out_dtype not in (None, "float32"): raise ValueError("Apple GEMM 只支持 float32")
    if accumulate and out is None: raise ValueError("累加 GEMM 需要提供输出存储")
    if transpose_left: a = apple.transpose(a, (inner, rows))
    if transpose_right: b = apple.transpose(b, (cols, inner))
    result = apple.matmul(a, b, (rows, inner), (inner, cols))
    if out is None: return result
    apple.copy_into(out, apple.add(out, result) if accumulate else result)
    return out


def take(values, start, size):
    return apple.gather(values, range(start, start + size))


def add_into(target, source):
    apple.copy_into(target, apple.add(target, source))


def embedding(weight, ids, width):
    return apple.gather(weight, [token * width + j for token in ids for j in range(width)])


def embedding_backward(gradient, ids, count, width):
    return apple._run("lm_embedding_grad", count*width, gradient, apple.storage(ids, dtype="int64"), meta=(len(ids), width))


def rms_norm(values, weight, width, eps):
    shape = (values.size // width, width)
    average = apple.mean(apple.square(values), shape, 1)
    inverse = apple._run("math_unary", shape[0], apple.add(average, apple.full(shape[0], eps)), meta=(4,))
    normalized = apple.multiply(values, apple.broadcast(inverse, (shape[0], 1), shape))
    return apple.multiply(normalized, apple.broadcast(weight, (width,), shape)), inverse


def rms_norm_backward(gradient, values, weight, inverse, width):
    shape = (values.size // width, width)
    inv = apple.broadcast(inverse, (shape[0], 1), shape)
    weighted = apple.multiply(gradient, apple.broadcast(weight, (width,), shape))
    projection = apple.broadcast(apple.mean(apple.multiply(weighted, values), shape, 1), (shape[0], 1), shape)
    incoming = apple.multiply(inv, apple.subtract(weighted, apple.multiply(apple.multiply(values, apple.square(inv)), projection)))
    return incoming, apple.sum(apple.multiply(apple.multiply(gradient, values), inv), shape, 0)


def silu(values):
    return apple._run("lm_silu", values.size, values, meta=(0,))


def silu_backward(gradient, values):
    return apple._run("lm_silu", values.size, values, gradient, meta=(1,))


def rope(values, batch, time, heads, dim, theta, offset=0, inverse=False):
    return apple._run("lm_rope", values.size, values, meta=(time, heads, dim, offset, int(inverse)), scalars=(theta,))


def _heads(values, batch, time, heads, dim):
    return apple.transpose(values, (batch, time, heads, dim), (0, 2, 1, 3))


def _kv_positions(batch, time, heads, kv_heads, dim):
    return [((b*time+t)*kv_heads+h//(heads//kv_heads))*dim+j
            for b in range(batch) for h in range(heads) for t in range(time) for j in range(dim)]


def attention(q, k, v, batch, time, heads, kv_heads, dim):
    shape = (batch, heads, time, dim)
    query = _heads(q, batch, time, heads, dim)
    positions = _kv_positions(batch, time, heads, kv_heads, dim)
    key, value = apple.gather(k, positions), apple.gather(v, positions)
    scores = apple.scale(apple.matmul(query, apple.transpose(key, shape, (0, 1, 3, 2)), shape, (batch, heads, dim, time)), 1/sqrt(dim))
    mask = apple.storage(0 if j <= i else -float('inf') for i in range(time) for j in range(time))
    scores = apple.add(scores, apple.broadcast(mask, (time, time), (batch, heads, time, time)))
    weights = apple.softmax(scores, (batch, heads, time, time), 3)
    output = apple.matmul(weights, value, (batch, heads, time, time), shape)
    return apple.transpose(output, shape, (0, 2, 1, 3)), weights


def attention_backward(gradient, q, k, v, out, weights, batch, time, heads, kv_heads, dim):
    shape, score_shape = (batch, heads, time, dim), (batch, heads, time, time)
    positions = _kv_positions(batch, time, heads, kv_heads, dim)
    query, incoming = _heads(q, batch, time, heads, dim), _heads(gradient, batch, time, heads, dim)
    key, value = apple.gather(k, positions), apple.gather(v, positions)
    dp = apple.matmul(incoming, apple.transpose(value, shape, (0, 1, 3, 2)), shape, (batch, heads, dim, time))
    center = apple.broadcast(apple.sum(apple.multiply(dp, weights), score_shape, 3), (batch, heads, time, 1), score_shape)
    ds = apple.scale(apple.multiply(weights, apple.subtract(dp, center)), 1/sqrt(dim))
    dq = apple.matmul(ds, key, score_shape, shape)
    dk = apple.matmul(apple.transpose(ds, score_shape, (0, 1, 3, 2)), query, score_shape, shape)
    dv = apple.matmul(apple.transpose(weights, score_shape, (0, 1, 3, 2)), incoming, score_shape, shape)
    grouped = (batch, kv_heads, heads//kv_heads, time, dim)
    dk = apple.transpose(apple.sum(dk, grouped, 2), (batch, kv_heads, time, dim), (0, 2, 1, 3))
    dv = apple.transpose(apple.sum(dv, grouped, 2), (batch, kv_heads, time, dim), (0, 2, 1, 3))
    return apple.transpose(dq, shape, (0, 2, 1, 3)), dk, dv


def routing(scores, rows, experts, top_k):
    capacity = rows * top_k
    probs = apple.softmax(scores, (rows, experts), 1)
    indices = apple._run("lm_route_select", capacity, probs, meta=(experts, top_k), dtype="int64")
    weights = apple._run("lm_route_weights", capacity, probs, indices, meta=(experts, top_k))
    counts = apple._run("lm_route_counts", experts, indices, meta=(capacity,), dtype="int64")
    locations = apple._run("lm_route_pack", experts, indices, meta=(capacity,), out=apple.AppleStorage(size=experts*capacity, dtype="int64"))
    utilization = apple._run("lm_route_util", experts, counts, scalars=(1/capacity,), dtype="float32")
    aux = apple.scale(apple.sum(apple.multiply(utilization, apple.mean(probs, (rows, experts), 0))), experts)
    return dict(probs=probs, weights=weights, indices=indices, counts=tuple(counts.tolist()), locations=locations,
                utilization=utilization, aux=aux, rows=rows, experts=experts, top_k=top_k, capacity=capacity)


def routing_backward(gradient, route, auxiliary=False):
    rows, experts, top_k = (route[n] for n in ("rows", "experts", "top_k"))
    if auxiliary:
        return apple._run("lm_route_aux_grad", rows*experts, gradient, route['probs'], route['utilization'], meta=(rows, experts))
    return apple._run("lm_route_grad", rows*experts, gradient, route['weights'], route['indices'], meta=(experts, top_k))


def route_gather(values, route, expert, width):
    return apple._run("lm_route_gather", route['counts'][expert]*width, values, route['locations'], meta=(expert*route['capacity'], width, route['top_k']))


def route_gather_backward(gradient, route, expert, width):
    return apple._run("lm_route_scatter", gradient.size, gradient, route['locations'], meta=(expert*route['capacity'], width, route['top_k']), out=apple.full(route['rows']*width))


def combine(outputs, weights, route, width):
    result = apple.full(route['rows']*width)
    for expert, values in outputs:
        apple._run("lm_combine", values.size, values, weights, route['locations'], meta=(expert*route['capacity'], width, route['top_k']), out=result)
    return result


def combine_backward(gradient, outputs, weights, route, width):
    grads, dw = [], apple.full(weights.size)
    for expert, values in outputs:
        meta=(expert*route['capacity'], width, route['top_k'])
        grads.append(apple._run("lm_combine_grad", values.size, gradient, weights, route['locations'], meta=meta))
        apple._run("lm_combine_weight_grad", values.size//width, gradient, values, route['locations'], meta=meta, out=dw)
    return (*grads, dw)


def head_rows(values, start, count, width, time):
    return apple.gather(values, [((row//(time-1))*time+row%(time-1))*width+j for row in range(start,start+count) for j in range(width)])


def head_rows_backward(target, gradient, start, count, width, time):
    positions=[((row//(time-1))*time+row%(time-1))*width+j for row in range(start,start+count) for j in range(width)]
    add_into(target, apple.scatter(gradient, positions, target.size))


def prepare_labels(labels):
    return apple.storage(labels, dtype="int64")


def head_loss(logits, labels, rows, vocab, *, start=0):
    losses=apple._run("lm_head_loss", rows, logits, labels, meta=(vocab,start))
    return apple.sum(losses)


def head_gradient(logits, labels, rows, vocab, gradient, count, *, start=0):
    return apple._run("lm_head_grad", logits.size, apple.softmax(logits, (rows,vocab), 1), labels, gradient, meta=(vocab,start,count))
