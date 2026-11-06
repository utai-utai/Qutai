"""批量量子层与共同实数反向引擎之间的内部连接。"""

from functools import partial

from ..backends import cuda_quantum
from ..core.context import is_recording
from ..core.grad import record
from ..core.tensor import Tensor


def pqc(inputs: Tensor, weight: Tensor, rotations, entangler) -> Tensor:
    tracing = is_recording() and (inputs.requires_grad or weight.requires_grad)
    x, w = (
        (inputs.storage.copy(), weight.storage.copy())
        if tracing
        else (inputs.storage, weight.storage)
    )
    values = cuda_quantum.pqc_forward(x, w, rotations, entangler)
    rule = partial(
        cuda_quantum.pqc_backward,
        inputs=x,
        weights=w,
        rotations=rotations,
        entangler=entangler,
        input_grad=inputs.requires_grad,
        weight_grad=weight.requires_grad,
    )
    return record(Tensor(values, shape=inputs.shape), (inputs, weight), rule)
