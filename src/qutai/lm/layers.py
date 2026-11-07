"""解码器的专用运算与层；前向和局部反向保持在一起。"""

from math import prod, sqrt
from ..backends import cpu_lm, cuda_lm
from ..backends.device import get_backend
from ..backends.precision import matmul_precision, set_matmul_precision
from ..core.grad import record
from ..core.tensor import Tensor


def backend(value):
    device = value if isinstance(value, str) else value.device
    if device == "cpu":
        return cpu_lm
    if device == "cuda":
        return cuda_lm
    raise ValueError("此设备尚未实现大型语言模型内核")


def activation_dtype(x, weight):
    """BF16 只保存 CUDA 激活；模型参数和其梯度继续使用原精度。"""
    if (
        x.device == "cuda"
        and weight.dtype == "float32"
        and (matmul_precision() == "bfloat16" or x.dtype == "bfloat16")
    ):
        return "bfloat16"
    return weight.dtype


def residual_add(x, update):
    return x + update.to(dtype=x.dtype)


def linear(x, weight):
    b = backend(x)
    rows = x.size // x.shape[-1]
    out_features, inner = weight.shape
    precision = matmul_precision()
    values = b.gemm(
        x.storage,
        weight.storage,
        rows,
        inner,
        out_features,
        transpose_right=True,
        out_dtype=activation_dtype(x, weight),
    )

    def backward(g):
        with set_matmul_precision(precision):
            dx = (
                b.gemm(g, weight.storage, rows, out_features, inner, out_dtype=x.dtype)
                if x.requires_grad
                else None
            )
            dw = (
                b.gemm(
                    g,
                    x.storage,
                    out_features,
                    rows,
                    inner,
                    transpose_left=True,
                    out_dtype=weight.dtype,
                )
                if weight.requires_grad
                else None
            )
        return (dx, dw)

    return record(
        Tensor(values, shape=(*x.shape[:-1], out_features)), (x, weight), backward
    )


def embed(ids, weight, shape):
    b = backend(weight)
    width = weight.shape[1]
    values = b.embedding(weight.storage, ids, width)
    return record(
        Tensor(values, shape=(*shape, width)),
        (weight,),
        lambda g: (b.embedding_backward(g, ids, weight.shape[0], width),),
    )


def rms_norm(x, weight, eps):
    b = backend(x)
    width = x.shape[-1]
    out, inverse = b.rms_norm(x.storage, weight.storage, width, eps)
    return record(
        Tensor(out, shape=x.shape),
        (x, weight),
        lambda g: b.rms_norm_backward(g, x.storage, weight.storage, inverse, width),
    )


def silu(x):
    b = backend(x)
    return record(
        Tensor(b.silu(x.storage), shape=x.shape),
        (x,),
        lambda g: (b.silu_backward(g, x.storage),),
    )


def rope(x, heads, theta, offset=0):
    b = backend(x)
    batch, time, features = x.shape
    dims = (batch, time, heads, features // heads, theta, offset)
    return record(
        Tensor(b.rope(x.storage, *dims), shape=x.shape),
        (x,),
        lambda g: (b.rope(g, *dims, inverse=True),),
    )


def attention(q, k, v, heads, kv_heads):
    b = backend(q)
    batch, time, features = q.shape
    dims = (batch, time, heads, kv_heads, features // heads)
    out, lse = b.attention(q.storage, k.storage, v.storage, *dims)
    return record(
        Tensor(out, shape=q.shape),
        (q, k, v),
        lambda g: b.attention_backward(
            g, q.storage, k.storage, v.storage, out, lse, *dims
        ),
    )


def route_gather(x, route, expert):
    b = backend(x)
    width = x.shape[-1]
    out = b.route_gather(x.storage, route, expert, width)
    return record(
        Tensor(out, shape=(route["counts"][expert], width)),
        (x,),
        lambda g: (b.route_gather_backward(g, route, expert, width),),
    )


def combine(outputs, weights, route, width):
    b = backend(weights)
    saved = tuple(((e, y.storage) for e, y in outputs))
    values = b.combine(saved, weights.storage, route, width)
    return record(
        Tensor(values, shape=(route["rows"], width)),
        tuple((y for _, y in outputs)) + (weights,),
        lambda g: b.combine_backward(g, saved, weights.storage, route, width),
    )


def pack(hidden, aux):
    base = get_backend(hidden.device)
    b = backend(hidden)
    values = base.concatenate((hidden.storage, aux.storage), ((hidden.size,), (1,)))
    return record(
        Tensor(values, shape=(hidden.size + 1,)),
        (hidden, aux),
        lambda g: (b.take(g, 0, hidden.size), b.take(g, hidden.size, 1)),
    )


def unpack(packed, shape):
    base = get_backend(packed.device)
    b = backend(packed)
    size = prod(shape)
    hidden = record(
        Tensor(b.take(packed.storage, 0, size), shape=shape),
        (packed,),
        lambda g: (
            base.concatenate((g, base.full(1, dtype=g.dtype)), ((size,), (1,))),
        ),
    )
    aux = record(
        Tensor(b.take(packed.storage, size, 1), shape=()),
        (packed,),
        lambda g: (
            base.concatenate((base.full(size, dtype=g.dtype), g), ((size,), (1,))),
        ),
    )
    return (hidden, aux)


def projection_loss(hidden, weight, labels, chunk_size, *, prepared_labels=None):
    b = backend(hidden)
    base = get_backend(hidden.device)
    batch, time, width = hidden.shape
    vocab = weight.shape[0]
    precision = matmul_precision()
    targets = tuple((label for row in labels for label in row[1:]))
    label_data = (
        b.prepare_labels(targets) if prepared_labels is None else prepared_labels
    )
    count = max(1, sum((label != -100 for label in targets)))
    result = base.full(1, dtype=hidden.dtype)
    for start in range(0, len(targets), chunk_size):
        part = targets[start : start + chunk_size]
        rows = len(part)
        inputs = b.head_rows(hidden.storage, start, rows, width, time)
        logits = b.gemm(
            inputs,
            weight.storage,
            rows,
            width,
            vocab,
            transpose_right=True,
            out_dtype=activation_dtype(hidden, weight),
        )
        b.add_into(result, b.head_loss(logits, label_data, rows, vocab, start=start))
    result = base.scale(result, 1 / count)

    def backward(gradient):
        with set_matmul_precision(precision):
            dx = base.full(hidden.size, dtype=hidden.dtype)
            dw = base.full(weight.size, dtype=weight.dtype)
            for start in range(0, len(targets), chunk_size):
                part = targets[start : start + chunk_size]
                rows = len(part)
                inputs = b.head_rows(hidden.storage, start, rows, width, time)
                logits = b.gemm(
                    inputs,
                    weight.storage,
                    rows,
                    width,
                    vocab,
                    transpose_right=True,
                    out_dtype=activation_dtype(hidden, weight),
                )
                incoming = b.head_gradient(
                    logits, label_data, rows, vocab, gradient, count, start=start
                )
                if hidden.requires_grad:
                    contribution = b.gemm(
                        incoming,
                        weight.storage,
                        rows,
                        vocab,
                        width,
                        out_dtype=hidden.dtype,
                    )
                    b.head_rows_backward(dx, contribution, start, rows, width, time)
                if weight.requires_grad:
                    b.gemm(
                        incoming,
                        inputs,
                        vocab,
                        rows,
                        width,
                        transpose_left=True,
                        out_dtype=weight.dtype,
                        out=dw,
                        accumulate=True,
                    )
            return (dx, dw)

    return record(Tensor(result, shape=()), (hidden, weight), backward)


class Projection:
    def __init__(self, inputs, outputs, *, device, dtype, seed, std=0.02):
        self.weight = Tensor(
            backend(device).normal(inputs * outputs, std=std, seed=seed, dtype=dtype),
            shape=(outputs, inputs),
            requires_grad=True,
        )

    def __call__(self, x):
        return linear(x, self.weight)

    def parameters(self):
        return (self.weight,)


class RMSNorm:
    def __init__(self, width, eps, *, device, dtype):
        self.weight = Tensor(
            get_backend(device).full(width, 1, dtype=dtype),
            shape=(width,),
            requires_grad=True,
        )
        self.eps = eps

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)

    def parameters(self):
        return (self.weight,)


class SwiGLU:
    def __init__(self, width, hidden, *, device, dtype, seed, std=0.02):
        self.gate = Projection(width, hidden, device=device, dtype=dtype, seed=seed)
        self.up = Projection(
            width,
            hidden,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + 1,
        )
        self.down = Projection(
            hidden,
            width,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + 2,
            std=std,
        )

    def __call__(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))

    def parameters(self):
        return (*self.gate.parameters(), *self.up.parameters(), *self.down.parameters())


class GQA:
    def __init__(self, config, *, device, dtype, seed):
        kw = dict(device=device, dtype=dtype)
        width = config.d_model
        kv_width = config.n_kv_heads * (width // config.n_heads)
        self.q = Projection(width, width, seed=seed, **kw)
        self.k = Projection(
            width, kv_width, seed=None if seed is None else seed + 1, **kw
        )
        self.v = Projection(
            width, kv_width, seed=None if seed is None else seed + 2, **kw
        )
        self.out = Projection(
            width,
            width,
            seed=None if seed is None else seed + 3,
            std=0.02 / sqrt(2 * config.n_layers),
            **kw,
        )
        self.heads, self.kv_heads, self.theta = (
            config.n_heads,
            config.n_kv_heads,
            config.rope_theta,
        )

    def __call__(self, x):
        q = rope(self.q(x), self.heads, self.theta)
        k = rope(self.k(x), self.kv_heads, self.theta)
        v = self.v(x)
        return self.out(attention(q, k, v, self.heads, self.kv_heads))

    def parameters(self):
        return tuple(
            (
                p
                for layer in (self.q, self.k, self.v, self.out)
                for p in layer.parameters()
            )
        )


class SparseMoE:
    def __init__(self, config, *, device, dtype, seed):
        self.router = Projection(
            config.d_model, config.n_experts, device=device, dtype=dtype, seed=seed
        )
        self.experts = tuple(
            (
                SwiGLU(
                    config.d_model,
                    config.d_expert,
                    device=device,
                    dtype=dtype,
                    seed=None if seed is None else seed + 1 + 3 * i,
                    std=0.02 / sqrt(2 * config.n_layers),
                )
                for i in range(config.n_experts)
            )
        )
        self.top_k, self.n_experts = (config.top_k, config.n_experts)
        self.utilization = None
        self.last_counts = (0,) * self.n_experts

    def __call__(self, x):
        with set_matmul_precision(None):
            scores = self.router(x.to(dtype=self.router.weight.dtype))
        b = backend(x)
        rows = x.size // x.shape[-1]
        route = b.routing(scores.storage, rows, self.n_experts, self.top_k)
        weights = record(
            Tensor(route["weights"], shape=(rows, self.top_k)),
            (scores,),
            lambda g: (b.routing_backward(g, route),),
        )
        aux = record(
            Tensor(route["aux"], shape=()),
            (scores,),
            lambda g: (b.routing_backward(g, route, True),),
        )
        self.last_counts = route["counts"]
        self.utilization = Tensor(route["utilization"], shape=(self.n_experts,))
        outputs = []
        for expert, layer in enumerate(self.experts):
            if route["counts"][expert]:
                outputs.append((expert, layer(route_gather(x, route, expert))))
        output = combine(outputs, weights, route, x.shape[-1])
        result = record(
            Tensor(output.storage, shape=x.shape), (output,), lambda g: (g,)
        )
        return (result, aux)

    def parameters(self):
        return (
            *self.router.parameters(),
            *(p for expert in self.experts for p in expert.parameters()),
        )


class Block:
    def __init__(self, config, *, device, dtype, seed):
        kw = dict(device=device, dtype=dtype)
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps, **kw)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps, **kw)
        self.attn = GQA(config, seed=seed, **kw)
        self.is_moe = config.ffn_type == "moe"
        self.ffn = (
            SparseMoE(config, seed=None if seed is None else seed + 4, **kw)
            if self.is_moe
            else SwiGLU(
                config.d_model,
                config.d_ff,
                seed=None if seed is None else seed + 4,
                std=0.02 / sqrt(2 * config.n_layers),
                **kw,
            )
        )

    def __call__(self, x):
        x = residual_add(x, self.attn(self.attn_norm(x)))
        if self.is_moe:
            y, aux = self.ffn(self.ffn_norm(x))
        else:
            y = self.ffn(self.ffn_norm(x))
            aux = Tensor(get_backend(x.device).full(1, dtype=x.dtype), shape=())
        return (residual_add(x, y), aux)

    def hidden(self, x):
        """Dense 重计算只保留 hidden，不打包没有梯度的零辅助损失。"""
        return self(x)[0]

    def packed(self, x):
        return pack(*self(x))

    def parameters(self):
        return tuple(
            (
                p
                for layer in (self.attn_norm, self.ffn_norm, self.attn, self.ffn)
                for p in layer.parameters()
            )
        )
