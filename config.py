from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    ffn_dim: int = 1536
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 16
    grad_accum_steps: int = 4
    epochs: float = 5.0
    token_budget: int = 60_000_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 32
    save_interval: int = 250


def config_dict() -> dict:
    return {"model": asdict(ModelConfig()), "train": asdict(TrainConfig())}
