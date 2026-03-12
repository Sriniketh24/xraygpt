"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "iu_xray"
    data_dir: str = "data/raw/iu_xray"
    processed_dir: str = "data/processed/iu_xray"
    image_size: int = 224
    max_length: int = 128
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    view: str = "frontal"


@dataclass
class VisionConfig:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    hidden_size: int = 768
    freeze: bool = True
    num_prefix_tokens: int = 8


@dataclass
class ProjectionConfig:
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class TextConfig:
    model_name: str = "gpt2"
    freeze_embeddings: bool = False
    max_new_tokens: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 30
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True
    save_every: int = 5
    eval_every: int = 1
    patience: int = 7


@dataclass
class GenerationConfig:
    strategy: str = "greedy"
    beam_size: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.2


@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_dir: str = "outputs/logs"


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project_name: str = "xraygpt"
    log_every: int = 50


@dataclass
class XRayGPTConfig:
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> XRayGPTConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    config = XRayGPTConfig()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return config

    # Update each sub-config from the YAML
    section_map = {
        "data": (config.data, DataConfig),
        "vision": (config.vision, VisionConfig),
        "projection": (config.projection, ProjectionConfig),
        "text": (config.text, TextConfig),
        "training": (config.training, TrainingConfig),
        "generation": (config.generation, GenerationConfig),
        "paths": (config.paths, PathsConfig),
        "logging": (config.logging, LoggingConfig),
    }

    for section_name, (section_obj, _) in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            for key, value in raw[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    return config
