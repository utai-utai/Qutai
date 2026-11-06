"""Dense/MoE 解码器配置；参数预算不需要先分配模型。"""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass
class Config:
    vocab_size: int = 32768
    n_layers: int = 20
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 3584
    max_seq_len: int = 1024
    ffn_type: str = "dense"
    n_experts: int = 8
    top_k: int = 2
    d_expert: int = 1792
    rope_theta: float = 10000.0
    norm_eps: float = 1e-06
    aux_loss_coef: float = 0.01
    activation_checkpointing: bool = True

    def __post_init__(self):
        dimensions = (
            self.vocab_size,
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.d_ff,
            self.max_seq_len,
            self.n_experts,
            self.d_expert,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("模型维度必须为正整数")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("GQA 特征数与头数不匹配")
        if self.d_model // self.n_heads % 2 or self.d_model // self.n_heads > 256:
            raise ValueError("RoPE 每头维度必须为不超过 256 的偶数")
        if (
            self.ffn_type not in ("dense", "moe")
            or type(self.top_k) is not int
            or not 1 <= self.top_k <= self.n_experts
        ):
            raise ValueError("FFN 类型或专家数量无效")
        settings = (self.norm_eps, self.rope_theta, self.aux_loss_coef)
        if any(
            not isinstance(value, (int, float)) or not isfinite(value)
            for value in settings
        ):
            raise ValueError("归一化、RoPE 和辅助损失系数必须为有限数值")
        if self.norm_eps <= 0 or self.rope_theta <= 0 or self.aux_loss_coef < 0:
            raise ValueError("归一化、RoPE 或辅助损失配置无效")

    def to_dict(self):
        return asdict(self)

    def parameter_counts(self):
        d, k = (self.d_model, self.n_kv_heads * (self.d_model // self.n_heads))
        shared = self.vocab_size * d + d
        attention = 2 * d * d + 2 * d * k + 2 * d
        if self.ffn_type == "dense":
            total = shared + self.n_layers * (attention + 3 * d * self.d_ff)
            return {"total_parameters": total, "active_parameters": total}
        expert = 3 * d * self.d_expert
        base = shared + self.n_layers * (attention + d * self.n_experts)
        return {
            "total_parameters": base + self.n_layers * self.n_experts * expert,
            "active_parameters": base + self.n_layers * self.top_k * expert,
        }
