from ..backends.device import get_backend
from ..backends.storage import Storage
from .context import is_recording
from .types import Backward, GradNode


def record[T: GradNode](result: T, parents: tuple[T, ...], rule: Backward | None = None) -> T:
    """结果保存输入引用和已绑定的局部反向规则。"""
    result.requires_grad = is_recording() and any(parent.requires_grad for parent in parents)
    if result.requires_grad:
        result.parents = parents
        result.backward_rule = rule
    return result


def _topological_order[T: GradNode](output: T) -> list[T]:
    """迭代遍历计算图，输入在前、结果在后，每个节点只出现一次。"""
    order: list[T] = []
    seen: set[T] = set()
    pending = [(output, False)]

    while pending:
        tensor, expanded = pending.pop()
        if expanded: order.append(tensor)
        elif tensor not in seen:
            seen.add(tensor)
            if getattr(tensor, "_graph_released", False): raise RuntimeError("计算图已释放；请重新前向，或首次 backward 时保留计算图")
            if tensor.parents and tensor.backward_rule is None: raise NotImplementedError("计算图包含尚未实现反向规则的运算")
            pending.append((tensor, True))
            pending.extend((parent, False) for parent in tensor.parents if parent.requires_grad)
    return order


def backward[T: GradNode](output: T, gradient: Storage, *, retain_graph: bool = True) -> dict[T, Storage]:
    """沿计算图传播梯度存储，返回各叶子节点的梯度。"""
    order = _topological_order(output)
    gradients = {output: gradient}
    leaves: dict[T, Storage] = {}
    while order:
        tensor = order.pop()
        incoming = gradients.pop(tensor, None)
        if incoming is None: continue
        rule = tensor.backward_rule
        if rule is None:
            leaves[tensor] = incoming
            continue
        contributions = rule(incoming)
        parents = tensor.parents
        if not retain_graph:
            tensor.parents = ()
            tensor.backward_rule = None
            tensor._graph_released = True
        for parent, contribution in zip(parents, contributions, strict=True):
            if parent.requires_grad and contribution is not None:
                gradients[parent] = (get_backend(contribution.device).add(gradients[parent], contribution) if parent in gradients else contribution)
        del rule, parents, contributions
    return leaves
