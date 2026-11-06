"""共享电路计划的批量 PQC；按 token 分块，状态向量不进入持久计算图。"""

from functools import lru_cache

from .cuda import CUDAStorage, _launch, _p, _i, full, storage
from .cuda_runtime import get_runtime


@lru_cache(maxsize=64)
def _plan(rotations, entangler):
    qubits = len(rotations[0])
    gates = [(1, wire, 0, wire) for wire in range(qubits)]
    for layer, names in enumerate(rotations):
        gates.extend(
            ({"RX": 0, "RY": 1, "RZ": 2}[name], wire, 1, layer * qubits + wire)
            for wire, name in enumerate(names)
        )
        gates.extend(
            (3 if entangler == "cnot" else 4, wire + 1, wire, 0)
            for wire in range(qubits - 1)
        )
    gates.extend((1, wire, 2, 0) for wire in range(qubits))
    return tuple(gates)


def _device_plan(rotations, entangler):
    return _plan_storage(rotations, entangler, get_runtime())


@lru_cache(maxsize=64)
def _plan_storage(rotations, entangler, runtime):
    return storage(
        (value for gate in _plan(rotations, entangler) for value in gate), dtype="int64"
    )


def _chunks(rows, qubits, dtype, chunk_size):
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("量子批次分块大小必须是正整数")
    # 反向最多同时保留 ket、previous、bra、scratch 四份状态。
    budget_rows = max(
        1, (64 * 1024 * 1024) // (4 * (1 << qubits) * (8 if dtype == "float32" else 16))
    )
    chunk = min(chunk_size or budget_rows, budget_rows)
    for start in range(0, rows, chunk):
        yield start, min(chunk, rows - start)


def _gate(source, target, inputs, weights, plan, qubits, start, index, inverse=False):
    _launch(
        "qb_gate",
        source.size,
        _p(source),
        _p(target),
        _p(inputs),
        _p(weights),
        _p(plan),
        _i(source.size),
        _i(1 << qubits),
        _i(qubits),
        _i(start),
        _i(index),
        _i(inverse),
        dtype=source.dtype,
        source="quantum_batch.cu",
    )


def _forward_chunk(inputs, weights, plan, gates, qubits, start, count):
    dtype = "complex64" if inputs.dtype == "float32" else "complex128"
    state, scratch = (
        CUDAStorage(count * (1 << qubits), dtype),
        CUDAStorage(count * (1 << qubits), dtype),
    )
    _launch(
        "qb_basis",
        state.size,
        _p(state),
        _i(state.size),
        _i(1 << qubits),
        dtype=dtype,
        source="quantum_batch.cu",
    )
    for index in range(len(gates)):
        _gate(state, scratch, inputs, weights, plan, qubits, start, index)
        state, scratch = scratch, state
    return state, scratch


def pqc_forward(inputs, weights, rotations, entangler, *, chunk_size=None):
    qubits, gates = len(rotations[0]), _plan(rotations, entangler)
    if (inputs.device, inputs.dtype) != (weights.device, weights.dtype):
        raise ValueError("量子输入与权重必须使用相同设备和精度")
    rows, plan = inputs.size // qubits, _device_plan(rotations, entangler)
    result = CUDAStorage(inputs.size, inputs.dtype)
    for start, count in _chunks(rows, qubits, inputs.dtype, chunk_size):
        state, scratch = _forward_chunk(
            inputs, weights, plan, gates, qubits, start, count
        )
        _launch(
            "qb_readout",
            count * qubits * 256,
            _p(state),
            _p(result),
            _i(1 << qubits),
            _i(qubits),
            _i(start),
            dtype=inputs.dtype,
            source="quantum_batch.cu",
        )
    return result


def pqc_backward(
    gradient,
    inputs,
    weights,
    rotations,
    entangler,
    *,
    input_grad=True,
    weight_grad=True,
    chunk_size=None,
):
    qubits, gates = len(rotations[0]), _plan(rotations, entangler)
    rows, plan = inputs.size // qubits, _device_plan(rotations, entangler)
    dx = full(inputs.size, dtype=inputs.dtype) if input_grad else None
    dw = full(weights.size, dtype=weights.dtype) if weight_grad else None
    for start, count in _chunks(rows, qubits, inputs.dtype, chunk_size):
        ket, previous = _forward_chunk(
            inputs, weights, plan, gates, qubits, start, count
        )
        bra, scratch = (
            CUDAStorage(ket.size, ket.dtype),
            CUDAStorage(ket.size, ket.dtype),
        )
        _launch(
            "qb_bra",
            ket.size,
            _p(ket),
            _p(gradient),
            _p(bra),
            _i(ket.size),
            _i(1 << qubits),
            _i(qubits),
            _i(start),
            dtype=inputs.dtype,
            source="quantum_batch.cu",
        )
        for index in reversed(range(len(gates))):
            _gate(
                ket, previous, inputs, weights, plan, qubits, start, index, inverse=True
            )
            gate = gates[index]
            if gate[0] < 3 and (
                (gate[2] == 0 and input_grad) or (gate[2] == 1 and weight_grad)
            ):
                _launch(
                    "qb_gradient",
                    count * 256,
                    _p(bra),
                    _p(previous),
                    _p(inputs),
                    _p(weights),
                    _p(plan),
                    _p(dx),
                    _p(dw),
                    _i(1 << qubits),
                    _i(qubits),
                    _i(start),
                    _i(index),
                    _i(input_grad),
                    _i(weight_grad),
                    dtype=inputs.dtype,
                    source="quantum_batch.cu",
                )
            _gate(
                bra, scratch, inputs, weights, plan, qubits, start, index, inverse=True
            )
            bra, scratch, ket, previous = scratch, bra, previous, ket
        del ket, previous, bra, scratch
    return dx, dw
