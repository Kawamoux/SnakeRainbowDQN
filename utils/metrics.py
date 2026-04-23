from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int
    global_step: int
    score: int
    episode_reward: float
    steps: int
    mean_loss: float
    moving_average: float


class MetricsTracker:
    def __init__(self, moving_average_window: int, csv_path: Path, reset: bool = False) -> None:
        self.moving_average_window = moving_average_window
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.episode_indices: list[int] = []
        self.global_steps: list[int] = []
        self.scores: list[int] = []
        self.episode_rewards: list[float] = []
        self.losses: list[float] = []
        self.steps: list[int] = []
        self.moving_averages: list[float] = []
        self._score_window: deque[int] = deque(maxlen=moving_average_window)
        self.best_moving_average = float("-inf")

        self._ensure_csv_header(reset=reset)

    def log_episode(
        self,
        episode: int,
        global_step: int,
        score: int,
        episode_reward: float,
        steps: int,
        mean_loss: float | None,
    ) -> EpisodeMetrics:
        safe_loss = float(mean_loss) if mean_loss is not None else 0.0

        self.episode_indices.append(episode)
        self.global_steps.append(global_step)
        self.scores.append(score)
        self.episode_rewards.append(float(episode_reward))
        self.losses.append(safe_loss)
        self.steps.append(steps)
        self._score_window.append(score)

        moving_average = float(np.mean(self._score_window))
        self.moving_averages.append(moving_average)
        self.best_moving_average = max(self.best_moving_average, moving_average)

        metrics = EpisodeMetrics(
            episode=episode,
            global_step=global_step,
            score=score,
            episode_reward=float(episode_reward),
            steps=steps,
            mean_loss=safe_loss,
            moving_average=moving_average,
        )
        self._append_to_csv(metrics)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "moving_average_window": self.moving_average_window,
            "episode_indices": self.episode_indices,
            "global_steps": self.global_steps,
            "scores": self.scores,
            "episode_rewards": self.episode_rewards,
            "losses": self.losses,
            "steps": self.steps,
            "moving_averages": self.moving_averages,
            "best_moving_average": self.best_moving_average,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.episode_indices = list(state_dict.get("episode_indices", []))
        self.global_steps = list(state_dict.get("global_steps", []))
        self.scores = list(state_dict.get("scores", []))
        self.episode_rewards = list(state_dict.get("episode_rewards", []))
        self.losses = list(state_dict.get("losses", []))
        self.steps = list(state_dict.get("steps", []))
        self.moving_averages = list(state_dict.get("moving_averages", []))
        self.best_moving_average = float(state_dict.get("best_moving_average", float("-inf")))
        self._score_window = deque(self.scores[-self.moving_average_window :], maxlen=self.moving_average_window)
        self._ensure_csv_header(reset=True)
        for index, score in enumerate(self.scores):
            metrics = EpisodeMetrics(
                episode=self.episode_indices[index] if index < len(self.episode_indices) else index + 1,
                global_step=self.global_steps[index] if index < len(self.global_steps) else 0,
                score=score,
                episode_reward=self.episode_rewards[index],
                steps=self.steps[index],
                mean_loss=self.losses[index],
                moving_average=self.moving_averages[index],
            )
            self._append_to_csv(metrics)

    def _ensure_csv_header(self, reset: bool = False) -> None:
        if reset or not self.csv_path.exists():
            fieldnames = list(asdict(EpisodeMetrics(0, 0, 0, 0.0, 0, 0.0, 0.0)).keys())
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    def _append_to_csv(self, metrics: EpisodeMetrics) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(asdict(metrics).keys()))
            writer.writerow(asdict(metrics))
