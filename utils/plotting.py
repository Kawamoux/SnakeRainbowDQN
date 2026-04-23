from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from utils.metrics import MetricsTracker


def save_training_plots(metrics: MetricsTracker, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    episodes = list(range(1, len(metrics.scores) + 1))
    if not episodes:
        return

    plt.style.use("ggplot")
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(episodes, metrics.scores, label="Score", color="#2f855a", linewidth=1.5)
    axes[0].plot(
        episodes,
        metrics.moving_averages,
        label=f"Moving average ({metrics.moving_average_window})",
        color="#1a365d",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Score")
    axes[0].set_title("Training progress")
    axes[0].legend()

    axes[1].plot(episodes, metrics.episode_rewards, label="Episode reward", color="#c05621", linewidth=1.5)
    axes[1].plot(episodes, metrics.losses, label="Mean loss", color="#805ad5", linewidth=1.5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
