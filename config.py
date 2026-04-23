from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_DIR = ARTIFACTS_DIR / "logs"
PLOT_DIR = ARTIFACTS_DIR / "plots"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EnvConfig:
    grid_width: int = 15
    grid_height: int = 15
    cell_size: int = 32
    render_fps: int = 12
    # "rich" was the most stable learning setup in this project. The grid
    # mode stays available for experiments, but it is no longer the default.
    observation_mode: str = "rich"
    compact_observation_size: int = 11
    rich_observation_size: int = 30
    action_size: int = 3
    food_reward: float = 10.0
    death_penalty: float = -20.0
    step_penalty: float = -0.01
    closer_reward: float = 0.02
    farther_penalty: float = -0.02
    timeout_penalty: float = -20.0
    full_clear_reward: float = 10.0
    trap_penalty: float = 0.0
    trap_min_open_space_buffer: int = 2
    # Safety shaping is useful for diagnostics, but it made recent runs noisy.
    # Keep it opt-in so the base Rainbow signal remains simple and learnable.
    safety_shaping: bool = False
    safety_min_length: int = 12
    low_space_penalty_scale: float = 0.02
    low_space_penalty_max: float = -1.0
    tail_unreachable_penalty: float = -0.25
    max_steps_without_food_multiplier: int = 6

    @property
    def max_steps_without_food(self) -> int:
        return self.grid_width * self.grid_height * self.max_steps_without_food_multiplier

    @property
    def grid_only_observation_size(self) -> int:
        return self.grid_width * self.grid_height * 3 + 6

    @property
    def grid_food_path_feature_size(self) -> int:
        return 6

    @property
    def grid_hybrid_legacy_observation_size(self) -> int:
        return self.rich_observation_size + self.grid_only_observation_size

    @property
    def grid_observation_size(self) -> int:
        return self.grid_hybrid_legacy_observation_size + self.grid_food_path_feature_size

    @property
    def observation_size(self) -> int:
        if self.observation_mode == "compact":
            return self.compact_observation_size
        if self.observation_mode == "rich":
            return self.rich_observation_size
        if self.observation_mode == "grid":
            return self.grid_observation_size
        if self.observation_mode == "grid_hybrid_legacy":
            return self.grid_hybrid_legacy_observation_size
        if self.observation_mode == "grid_legacy":
            return self.grid_only_observation_size
        raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")


@dataclass
class AgentConfig:
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 128
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_update_interval: int = 1_000
    gradient_clip: float = 10.0
    noisy_std: float = 0.5
    num_atoms: int = 51
    # C51 support must cover discounted returns. With +10 per food, [-20, 20]
    # clips successful episodes too aggressively and flattens long-term value.
    v_min: float = -50.0
    v_max: float = 300.0
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    per_epsilon: float = 1e-5


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 400
    # 0 disables the script-level cap and lets the environment decide when
    # an episode ends. Snake endgames often need far more than 1000 steps.
    max_steps_per_episode: int = 0
    moving_average_window: int = 50
    log_interval: int = 10
    plot_interval: int = 10
    checkpoint_interval: int = 50
    deterministic_torch: bool = False
    device: str = field(default_factory=default_device)


@dataclass
class EvalConfig:
    seed: int = 42
    episodes: int = 5
    render_mode: str = "human"
    fps: int = 12


@dataclass
class PathsConfig:
    artifacts_dir: Path = ARTIFACTS_DIR
    checkpoints_dir: Path = CHECKPOINT_DIR
    logs_dir: Path = LOG_DIR
    plots_dir: Path = PLOT_DIR
    best_model_path: Path = CHECKPOINT_DIR / "best_model.pt"
    latest_checkpoint_path: Path = CHECKPOINT_DIR / "latest_checkpoint.pt"
    training_metrics_path: Path = LOG_DIR / "training_metrics.csv"
    training_plot_path: Path = PLOT_DIR / "training_curves.png"
    config_snapshot_path: Path = ARTIFACTS_DIR / "run_config.json"


@dataclass
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def build_config() -> ProjectConfig:
    return ProjectConfig()


def ensure_directories(paths: PathsConfig) -> None:
    for directory in (
        paths.artifacts_dir,
        paths.checkpoints_dir,
        paths.logs_dir,
        paths.plots_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return value.relative_to(ROOT_DIR).as_posix()
        except ValueError:
            return value.as_posix()
    if isinstance(value, dict):
        return {key: _serialize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    return _serialize(asdict(config))


def save_config_snapshot(config: ProjectConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2)
