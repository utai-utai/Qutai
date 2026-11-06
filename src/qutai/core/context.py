from contextvars import ContextVar, Token


_recording = ContextVar("qutai_grad_recording", default=True)
_training = ContextVar("qutai_nn_training", default=True)


def set_recording(enabled: bool) -> Token[bool]:
    """临时控制计算图记录；嵌套调用保留外层暂停记录的状态。"""
    return _recording.set(_recording.get() and enabled)


def is_recording() -> bool:
    """当前是否允许记录计算图。"""
    return _recording.get()


def set_training(enabled: bool) -> Token[bool]:
    """临时控制层的训练行为；嵌套调用保留外层的预测状态。"""
    return _training.set(_training.get() and enabled)


def is_training() -> bool:
    """当前是否使用训练行为，例如更新 BatchNorm 统计量。"""
    return _training.get()
