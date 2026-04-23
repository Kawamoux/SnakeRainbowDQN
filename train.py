from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
from tqdm import trange

from agent.rainbow_agent import RainbowAgent, peek_checkpoint_state_dim
from config import build_config, ensure_directories, save_config_snapshot
from env.snake_env import SnakeEnv
from utils.metrics import MetricsTracker
from utils.plotting import save_training_plots
from utils.seed import seed_env_spaces, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Rainbow DQN agent on Snake.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to resume from.",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint used only to initialize compatible weights "
            "for a fresh run. Optimizer and replay buffer are reset."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the compute device (cpu or cuda).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print action distribution and reward-component summaries every log interval.",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        choices=("compact", "rich", "grid"),
        default=None,
        help="Override the observation mode for a fresh run.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Run a deterministic evaluation every N episodes. Disabled with 0.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of deterministic evaluation episodes when eval-interval is enabled.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help=(
            "Optional script-level safety cap. Use 0 to disable it and let "
            "the environment end episodes naturally."
        ),
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=None,
        help="Save/update the training plot every N episodes. Use 0 to disable periodic plots.",
    )
    return parser.parse_args()


def infer_observation_mode_from_state_dim(config, state_dim: int) -> str:
    if state_dim == config.env.compact_observation_size:
        return "compact"
    if state_dim == config.env.rich_observation_size:
        return "rich"
    if state_dim == config.env.grid_observation_size:
        return "grid"
    if state_dim == config.env.grid_hybrid_legacy_observation_size:
        return "grid_hybrid_legacy"
    if state_dim == config.env.grid_only_observation_size:
        return "grid_legacy"
    raise ValueError(
        f"Unsupported checkpoint state_dim={state_dim}. "
        f"Expected {config.env.compact_observation_size}, "
        f"{config.env.rich_observation_size}, "
        f"{config.env.grid_observation_size}, "
        f"{config.env.grid_hybrid_legacy_observation_size}, or "
        f"{config.env.grid_only_observation_size}."
    )


def run_deterministic_evaluation(
    agent: RainbowAgent,
    config,
    episodes: int,
) -> dict[str, float]:
    eval_env = SnakeEnv(config.env, render_mode=None)
    scores: list[int] = []
    rewards: list[float] = []
    steps_hist: list[int] = []
    full_clears = 0
    collisions = 0
    timeouts = 0
    max_possible_score = config.env.grid_width * config.env.grid_height - 3

    for episode in range(1, episodes + 1):
        state, _ = eval_env.reset(seed=config.evaluation.seed + episode)
        terminated = False
        truncated = False
        info = {"score": 0}
        episode_reward = 0.0
        episode_steps = 0

        while not (terminated or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_steps += 1

        scores.append(int(info["score"]))
        rewards.append(float(episode_reward))
        steps_hist.append(episode_steps)
        if int(info["score"]) >= max_possible_score and not info.get("collision", False):
            full_clears += 1
        elif truncated:
            timeouts += 1
        else:
            collisions += 1

    eval_env.close()
    return {
        "score_mean": float(np.mean(scores)),
        "score_min": float(np.min(scores)),
        "score_p10": float(np.percentile(scores, 10)),
        "score_max": float(np.max(scores)),
        "reward_mean": float(np.mean(rewards)),
        "steps_mean": float(np.mean(steps_hist)),
        "full_clears": float(full_clears),
        "full_clear_rate": float(full_clears / max(episodes, 1)),
        "collisions": float(collisions),
        "timeouts": float(timeouts),
    }


def is_better_eval_score(candidate: dict[str, float], current: dict[str, float] | None) -> bool:
    if current is None:
        return True
    candidate_key = (
        candidate["full_clear_rate"],
        candidate["score_p10"],
        candidate["score_mean"],
        candidate["score_min"],
        candidate["score_max"],
    )
    current_key = (
        current["full_clear_rate"],
        current.get("score_p10", current["score_min"]),
        current["score_mean"],
        current["score_min"],
        current["score_max"],
    )
    return candidate_key >= current_key


def main() -> None:
    args = parse_args()
    config = build_config()

    if args.episodes is not None:
        config.train.episodes = args.episodes
    if args.seed is not None:
        config.train.seed = args.seed
        config.evaluation.seed = args.seed
    if args.device is not None:
        config.train.device = args.device
    if args.observation_mode is not None:
        config.env.observation_mode = args.observation_mode
    if args.max_steps_per_episode is not None:
        config.train.max_steps_per_episode = args.max_steps_per_episode
    if args.plot_interval is not None:
        config.train.plot_interval = args.plot_interval
    if config.train.max_steps_per_episode < 0:
        raise ValueError("--max-steps-per-episode must be >= 0.")
    if config.train.plot_interval < 0:
        raise ValueError("--plot-interval must be >= 0.")
    if args.checkpoint and args.warm_start_checkpoint:
        raise ValueError("--checkpoint and --warm-start-checkpoint cannot be used together.")

    if args.checkpoint:
        checkpoint_state_dim = peek_checkpoint_state_dim(args.checkpoint, device=config.train.device)
        inferred_mode = infer_observation_mode_from_state_dim(config, checkpoint_state_dim)
        config.env.observation_mode = inferred_mode

    ensure_directories(config.paths)
    save_config_snapshot(config, config.paths.config_snapshot_path)
    set_global_seed(config.train.seed, config.train.deterministic_torch)

    env = SnakeEnv(config.env, render_mode=None)
    seed_env_spaces(env, config.train.seed)

    agent = RainbowAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=config.env.action_size,
        config=config.agent,
        device=config.train.device,
    )
    metrics = MetricsTracker(
        moving_average_window=config.train.moving_average_window,
        csv_path=config.paths.training_metrics_path,
        reset=args.checkpoint is None,
    )

    start_episode = 1
    global_step = 0
    best_moving_average = float("-inf")
    best_eval_metrics: dict[str, float] | None = None
    warm_start_metadata: dict | None = None

    if args.checkpoint:
        metadata = agent.load(args.checkpoint, load_optimizer=True, load_buffer=True)
        tracker_state = metadata.get("metrics_tracker")
        if tracker_state:
            metrics.load_state_dict(tracker_state)
        start_episode = int(metadata.get("episode", 0)) + 1
        global_step = int(metadata.get("global_step", 0))
        best_moving_average = float(metadata.get("best_moving_average", float("-inf")))
        loaded_eval_metrics = metadata.get("best_eval_metrics")
        if isinstance(loaded_eval_metrics, dict):
            best_eval_metrics = {key: float(value) for key, value in loaded_eval_metrics.items()}
        print(f"Resumed from checkpoint: {args.checkpoint}")
    else:
        print(f"Starting new run with observation_mode={config.env.observation_mode}")
        if args.warm_start_checkpoint:
            warm_start_metadata = agent.warm_start_from_checkpoint(args.warm_start_checkpoint)
            print(
                "Warm-started from checkpoint: "
                f"{args.warm_start_checkpoint} | "
                f"source_dim={warm_start_metadata['source_state_dim']} | "
                f"target_dim={warm_start_metadata['target_state_dim']} | "
                f"copied_tensors={len(warm_start_metadata['copied_keys'])}"
            )

    progress_bar = trange(start_episode, config.train.episodes + 1, desc="Training", unit="episode")
    window_scores: list[int] = []
    window_rewards: list[float] = []
    window_losses: list[float] = []
    window_action_counts: Counter[int] = Counter()
    window_reward_components: defaultdict[str, float] = defaultdict(float)
    window_trap_entries = 0
    window_step_cap_hits = 0

    for episode in progress_bar:
        state, _ = env.reset(seed=config.train.seed + episode)
        episode_reward = 0.0
        episode_losses: list[float] = []
        steps_in_episode = 0
        info = {"score": 0}
        episode_action_counts: Counter[int] = Counter()
        episode_reward_components: defaultdict[str, float] = defaultdict(float)
        episode_trap_entries = 0
        previous_trap_state = False
        episode_step_cap_hit = False

        done = False
        while not done:
            if (
                config.train.max_steps_per_episode > 0
                and steps_in_episode >= config.train.max_steps_per_episode
            ):
                episode_step_cap_hit = True
                break

            action = agent.select_action(state, evaluate=False)
            episode_action_counts[action] += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            reward_breakdown = info.get("reward_breakdown", {})
            for name, value in reward_breakdown.items():
                episode_reward_components[name] += float(value)

            steps_in_episode += 1
            if (
                config.train.max_steps_per_episode > 0
                and steps_in_episode >= config.train.max_steps_per_episode
                and not (terminated or truncated)
            ):
                truncated = True
                episode_step_cap_hit = True
                info = dict(info)
                info["training_step_cap_reached"] = True

            done = terminated or truncated
            current_trap_state = bool(info.get("trap_detected", False))
            if current_trap_state and not previous_trap_state:
                episode_trap_entries += 1
            previous_trap_state = current_trap_state

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            global_step += 1

            if (
                global_step >= config.agent.learning_starts
                and global_step % config.agent.train_frequency == 0
            ):
                loss = agent.learn(global_step)
                if loss is not None:
                    episode_losses.append(loss)

            if global_step % config.agent.target_update_interval == 0:
                agent.update_target()

            if done:
                break

        agent.n_step_accumulator.clear()

        mean_loss = float(np.mean(episode_losses)) if episode_losses else None
        episode_metrics = metrics.log_episode(
            episode=episode,
            global_step=global_step,
            score=int(info["score"]),
            episode_reward=episode_reward,
            steps=steps_in_episode,
            mean_loss=mean_loss,
        )
        window_scores.append(episode_metrics.score)
        window_rewards.append(episode_metrics.episode_reward)
        if mean_loss is not None:
            window_losses.append(mean_loss)
        window_action_counts.update(episode_action_counts)
        for name, value in episode_reward_components.items():
            window_reward_components[name] += value
        window_trap_entries += episode_trap_entries
        window_step_cap_hits += int(episode_step_cap_hit)

        progress_bar.set_postfix(
            score=episode_metrics.score,
            moving_avg=f"{episode_metrics.moving_average:.2f}",
            loss=f"{episode_metrics.mean_loss:.4f}",
        )

        metadata = {
            "episode": episode,
            "global_step": global_step,
            "best_moving_average": best_moving_average,
            "best_eval_metrics": best_eval_metrics,
            "warm_start_metadata": warm_start_metadata,
            "metrics_tracker": metrics.state_dict(),
        }

        if episode_metrics.moving_average >= best_moving_average:
            best_moving_average = episode_metrics.moving_average
            metadata["best_moving_average"] = best_moving_average
            if args.eval_interval <= 0:
                agent.save(
                    config.paths.best_model_path,
                    metadata=metadata,
                    include_buffer=False,
                )

        if episode % config.train.checkpoint_interval == 0:
            agent.save(
                config.paths.latest_checkpoint_path,
                metadata=metadata,
                include_buffer=True,
            )

        if config.train.plot_interval > 0 and episode % config.train.plot_interval == 0:
            save_training_plots(metrics, config.paths.training_plot_path)

        if episode % config.train.log_interval == 0:
            print(
                f"Episode {episode:04d} | Score={episode_metrics.score} | "
                f"Reward={episode_metrics.episode_reward:.2f} | "
                f"MovingAvg={episode_metrics.moving_average:.2f} | "
                f"Loss={episode_metrics.mean_loss:.4f}"
            )
            if args.diagnostics and window_scores:
                total_actions = sum(window_action_counts.values())
                action_distribution = {
                    action: round(count / total_actions, 3) if total_actions else 0.0
                    for action, count in sorted(window_action_counts.items())
                }
                mean_reward_components = {
                    name: round(value / len(window_scores), 3)
                    for name, value in sorted(window_reward_components.items())
                }
                print(
                    "Diagnostics | "
                    f"ScoreMean={np.mean(window_scores):.2f} | "
                    f"ScoreMax={np.max(window_scores)} | "
                    f"RewardMean={np.mean(window_rewards):.2f} | "
                    f"LossMean={(np.mean(window_losses) if window_losses else 0.0):.4f} | "
                    "Exploration=NoisyNet | "
                    f"Actions={action_distribution} | "
                    f"TrapEntries={window_trap_entries} | "
                    f"StepCaps={window_step_cap_hits} | "
                    f"RewardBreakdown={mean_reward_components}"
                )
                window_scores.clear()
                window_rewards.clear()
                window_losses.clear()
                window_action_counts.clear()
                window_reward_components.clear()
                window_trap_entries = 0
                window_step_cap_hits = 0

        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_metrics = run_deterministic_evaluation(agent, config, args.eval_episodes)
            print(
                "DeterministicEval | "
                f"Episodes={args.eval_episodes} | "
                f"ScoreMean={eval_metrics['score_mean']:.2f} | "
                f"ScoreMin={eval_metrics['score_min']:.0f} | "
                f"ScoreP10={eval_metrics['score_p10']:.1f} | "
                f"ScoreMax={eval_metrics['score_max']:.0f} | "
                f"FullClears={eval_metrics['full_clears']:.0f}/{args.eval_episodes} | "
                f"Collisions={eval_metrics['collisions']:.0f} | "
                f"Timeouts={eval_metrics['timeouts']:.0f} | "
                f"RewardMean={eval_metrics['reward_mean']:.2f} | "
                f"StepsMean={eval_metrics['steps_mean']:.2f}"
            )
            if is_better_eval_score(eval_metrics, best_eval_metrics):
                best_eval_metrics = eval_metrics
                agent.save(
                    config.paths.best_model_path,
                    metadata={
                        "episode": episode,
                        "global_step": global_step,
                        "best_moving_average": best_moving_average,
                        "best_eval_metrics": best_eval_metrics,
                        "warm_start_metadata": warm_start_metadata,
                        "metrics_tracker": metrics.state_dict(),
                    },
                    include_buffer=False,
                )

    agent.save(
        config.paths.latest_checkpoint_path,
        metadata={
            "episode": config.train.episodes,
            "global_step": global_step,
            "best_moving_average": best_moving_average,
            "best_eval_metrics": best_eval_metrics,
            "warm_start_metadata": warm_start_metadata,
            "metrics_tracker": metrics.state_dict(),
        },
        include_buffer=True,
    )
    save_training_plots(metrics, config.paths.training_plot_path)
    env.close()

    print(f"Best model saved to: {config.paths.best_model_path}")
    print(f"Latest checkpoint saved to: {config.paths.latest_checkpoint_path}")
    print(f"Training metrics saved to: {config.paths.training_metrics_path}")
    print(f"Training plot saved to: {config.paths.training_plot_path}")


if __name__ == "__main__":
    main()
