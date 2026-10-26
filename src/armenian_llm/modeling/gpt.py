"""Небольшой decoder-only Transformer с RoPE и общими весами входа/выхода.

Все параметры инициализируются случайно. Предобученных компонентов нет.
Attention API: https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html
RoPE: https://arxiv.org/abs/2104.09864
SwiGLU: https://arxiv.org/abs/2002.05202
RMSNorm: https://arxiv.org/abs/1910.07467
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16_000
    context_length: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    ffn_hidden_dim: int = 1536
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in (
            "vocab_size", "context_length", "n_layers", "n_heads",
            "d_model", "ffn_hidden_dim",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if self.d_model % self.n_heads:
            raise ValueError("d_model должен делиться на n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("Размер головы должен быть чётным для RoPE")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout должен находиться в диапазоне [0, 1)")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta и norm_eps должны быть положительными")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Нормировку считаем в FP32 и при BF16 autocast.
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


def apply_rope(x, cos, sin):
    """Вращаем соседние пары координат Q или K; x имеет форму B,H,T,D."""
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.residual_dropout(self.out_proj(attended))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        head_dim = config.d_model // config.n_heads
        inverse_frequency = config.rope_theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.outer(
            torch.arange(config.context_length, dtype=torch.float32), inverse_frequency
        )
        # RoPE не добавляет обучаемых параметров; таблицы восстанавливаются из config.
        self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)

        self.apply(self._initialize_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, std=residual_std)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids, targets=None):
        """targets уже сдвинуты на один токен: x=tokens[:-1], y=tokens[1:]."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids должны иметь форму (batch, length)")
        length = input_ids.shape[1]
        if not 1 <= length <= self.config.context_length:
            raise ValueError("Длина входа должна находиться в пределах контекста")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets должны иметь ту же форму, что и input_ids")

        cos, sin = self.rope_cos[:, :, :length], self.rope_sin[:, :, :length]
        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        # Одна матрица служит и эмбеддингами входа, и выходной проекцией.
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss
