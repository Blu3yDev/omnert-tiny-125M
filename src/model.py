"""Decoder-only Omnert-Shakespeare Transformer, initialized from scratch."""
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, sequence, head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    even, odd = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (bsz, seq_len, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, seq_len, channels))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(config.d_model, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class OmnertShakespeare(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.training and getattr(self.config, "gradient_checkpointing", False):
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100) if targets is not None else None
        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_p=0.9, eos_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.context_length:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cutoff = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) - torch.softmax(sorted_logits, dim=-1)
            sorted_logits[cutoff > top_p] = -float("inf")
            filtered = torch.full_like(logits, -float("inf")).scatter(1, sorted_indices, sorted_logits)
            next_token = torch.multinomial(torch.softmax(filtered, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return input_ids

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
