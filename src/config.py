from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    competition_name: str = "nvidia-nemotron-model-reasoning-challenge"
    seed: int = 42

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")

    # Model
    base_model: str = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    kaggle_model: str = "metric/nemotron-3-nano-30b-a3b-bf16"

    # LoRA config
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 2048
    bf16: bool = True

    # Inference (from competition eval)
    max_tokens: int = 7680
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int = 8192

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.logs_dir = Path(self.logs_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def submissions_dir(self) -> Path:
        return self.output_dir / "submissions"

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "lora_adapter"
