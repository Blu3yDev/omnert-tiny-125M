from dataclasses import dataclass


@dataclass
class V2ModelConfig:
    vocab_size: int = 16000
    context_length: int = 512
    n_layers: int = 14
    n_heads: int = 12
    d_model: int = 768
    ffn_dim: int = 2496
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = True


@dataclass
class V2TrainConfig:
    seed: int = 2026
    microbatch_size: int = 4
    grad_accum_steps: int = 8
    token_budget: int = 600_000_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 24
    save_interval: int = 2000
    thermal_limit_c: int = 90
    source_weights: tuple[float, float, float] = (0.60, 0.25, 0.15)
