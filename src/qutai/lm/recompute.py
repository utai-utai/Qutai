"""确定性函数的激活重计算：前向只保留边界，反向重新建立局部计算图。"""

from collections.abc import Callable, Iterable
from functools import partial

from ..backends.device import get_backend
from ..backends.precision import matmul_precision, set_matmul_precision
from ..backends.storage import Storage
from ..core import grad
from ..core.context import _recording, set_recording
from ..core.tensor import Tensor


def checkpoint(
    function: Callable[..., Tensor],
    *arguments: Tensor,
    parameters: Iterable[Tensor] = (),
) -> Tensor:
    """重计算无随机性和副作用的函数；闭包模型权重通过 parameters 显式传入。"""
    parameters = tuple(dict.fromkeys(parameters))
    if any(not isinstance(value, Tensor) for value in (*arguments, *parameters)):
        raise TypeError("重计算的输入和参数必须是 Tensor")
    parents = tuple(dict.fromkeys((*arguments, *parameters)))
    with set_recording(False):
        output = function(*arguments)
    if not isinstance(output, Tensor):
        raise TypeError("重计算函数必须返回一个 Tensor")
    result = Tensor(output.storage, shape=output.shape)
    return grad.record(
        result,
        parents,
        partial(
            _backward,
            function=function,
            arguments=arguments,
            parameters=parameters,
            parents=parents,
            precision=matmul_precision(),
        ),
    )


def _backward(
    gradient: Storage, *, function, arguments, parameters, parents, precision
):
    # 不复制边界数据；局部 Tensor 身份独立，参数仍引用原模型叶子。
    replacements = {
        value: Tensor(
            value.storage, shape=value.shape, requires_grad=value.requires_grad
        )
        for value in arguments
    }
    with _recording.set(True), set_matmul_precision(precision):
        output = function(*(replacements[value] for value in arguments))
        if not isinstance(output, Tensor):
            raise TypeError("重计算函数必须返回一个 Tensor")
        leaves = (
            grad.backward(output, gradient, retain_graph=False)
            if output.requires_grad
            else {}
        )
    parameter_set = set(parameters)
    expected = set(replacements.values()) | parameter_set
    if any(value not in expected for value in leaves):
        raise ValueError(
            "重计算函数捕获了未声明的可训练 Tensor；请通过 parameters 显式传入"
        )
    results = []
    for value in parents:
        if not value.requires_grad:
            results.append(None)
            continue
        contribution = leaves.get(replacements.get(value, value))
        if value in replacements and value in parameter_set and value in leaves:
            contribution = (
                get_backend(value.device).add(contribution, leaves[value])
                if contribution is not None
                else leaves[value]
            )
        results.append(contribution)
    return tuple(results)
