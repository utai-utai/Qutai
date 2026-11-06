import json
from pathlib import Path

from ..nn.dit import DiT
from ..nn.model import Model as NeuralModel
from ..nn.serialization import describe_model, read_file
from .diffusion import Diffusion, FlowMatching


def save_model(network: DiT, method: Diffusion | FlowMatching, path: str | Path) -> None:
    """将网络、参数和生成配置写入一个 JSON 文件，不保存随机采样状态。"""
    data = describe_model(NeuralModel([network]), NeuralModel)
    data.update(generation=method.config(), value_range=[-1.0, 1.0])
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_method(path: str | Path) -> Diffusion | FlowMatching:
    """核对保存的采样规则，再恢复对应的生成方法。"""
    data = read_file(path)
    if not isinstance(data, dict) or "generation" not in data: raise ValueError("文件缺少生成方法配置")
    if data.get("value_range") != [-1.0, 1.0]: raise ValueError("生成模型的干净图像范围必须为 -1 到 1")
    configuration = data["generation"]
    if configuration["type"] == "Diffusion":
        if configuration["schedule"] != "cosine": raise ValueError("暂时只支持余弦噪声日程")
        return Diffusion(steps=configuration["steps"])
    if configuration["type"] == "FlowMatching":
        if configuration["solver"] != "euler": raise ValueError("暂时只支持 Euler 采样")
        return FlowMatching(steps=configuration["steps"])
    raise ValueError("文件中的生成方法不受支持")
