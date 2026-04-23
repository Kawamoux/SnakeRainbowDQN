from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agent.planner import SnakePlanner
from agent.rainbow_agent import RainbowAgent, peek_checkpoint_state_dim
from config import build_config
from env.snake_env import SnakeEnv
from utils.seed import seed_env_spaces, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Snake controllers.")
    parser.add_argument(
        "--controller",
        type=str,
        choices=("rainbow", "planner", "hybrid"),
        default="rainbow",
        help=(
            "rainbow uses the neural checkpoint, planner uses symbolic AI, "
            "hybrid lets Rainbow propose actions while the planner protects survival."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the Rainbow checkpoint. Defaults to the best saved model.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--device", type=str, default=None, help="Override the compute device.")
    parser.add_argument("--fps", type=int, default=None, help="Override the visual evaluation FPS.")
    parser.add_argument(
        "--observation-mode",
        type=str,
        choices=("compact", "rich", "grid"),
        default=None,
        help="Override the observation mode when not loading from a checkpoint.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the visual renderer and run evaluation without Pygame output.",
    )
    parser.add_argument(
        "--planner-shortcuts",
        action="store_true",
        help="Allow the planner to take safe shortest-path food shortcuts.",
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


def main() -> None:
    args = parse_args()
    config = build_config()

    if args.episodes is not None:
        config.evaluation.episodes = args.episodes
    if args.seed is not None:
        config.evaluation.seed = args.seed
    if args.device is not None:
        config.train.device = args.device
    if args.fps is not None:
        config.evaluation.fps = args.fps
    if args.observation_mode is not None:
        config.env.observation_mode = args.observation_mode
    config.env.render_fps = config.evaluation.fps

    needs_agent = args.controller in {"rainbow", "hybrid"}
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.paths.best_model_path
    if needs_agent:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_state_dim = peek_checkpoint_state_dim(checkpoint_path, device=config.train.device)
        config.env.observation_mode = infer_observation_mode_from_state_dim(config, checkpoint_state_dim)

    set_global_seed(config.evaluation.seed)
    render_mode = None if args.headless else config.evaluation.render_mode
    env = SnakeEnv(config.env, render_mode=render_mode)
    seed_env_spaces(env, config.evaluation.seed)

    agent: RainbowAgent | None = None
    if needs_agent:
        agent = RainbowAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=config.env.action_size,
            config=config.agent,
            device=config.train.device,
        )
        agent.load(checkpoint_path, load_optimizer=False, load_buffer=False)
        agent.set_eval_mode()

    planner: SnakePlanner | None = None
    if args.controller in {"planner", "hybrid"}:
        planner = SnakePlanner(
            width=config.env.grid_width,
            height=config.env.grid_height,
            allow_shortcuts=args.planner_shortcuts,
        )
        if not planner.has_cycle:
            print(
                "Planner warning: no Hamiltonian cycle for this grid; "
                "using BFS safety fallback only."
            )

    scores: list[int] = []
    steps_history: list[int] = []
    max_possible_score = config.env.grid_width * config.env.grid_height - 3
    full_clears = 0

    for episode in range(1, config.evaluation.episodes + 1):
        state, _ = env.reset(seed=config.evaluation.seed + episode)
        terminated = False
        truncated = False
        info = {"score": 0}
        steps = 0
        if render_mode is not None:
            env.render()

        while not (terminated or truncated):
            if args.controller == "planner":
                assert planner is not None
                action = planner.select_action(env)
            elif args.controller == "hybrid":
                assert agent is not None
                assert planner is not None
                preferred_action = (
                    agent.select_action(state, evaluate=True)
                    if args.planner_shortcuts
                    else None
                )
                action = planner.select_action(env, preferred_action=preferred_action)
            else:
                assert agent is not None
                action = agent.select_action(state, evaluate=True)

            state, _, terminated, truncated, info = env.step(action)
            steps += 1
            if render_mode is not None:
                env.render()

        scores.append(int(info["score"]))
        steps_history.append(steps)
        if int(info["score"]) >= max_possible_score and not info.get("collision", False):
            full_clears += 1
            end_reason = "full_clear"
        elif truncated:
            end_reason = "truncated"
        else:
            end_reason = "collision"
        print(
            f"Evaluation episode {episode:02d} | Controller={args.controller} | "
            f"Score={info['score']} | Steps={steps} | End={end_reason}"
        )

    env.close()

    print(f"Mean score: {np.mean(scores):.2f}")
    print(f"Max score: {np.max(scores)}")
    print(f"Min score: {np.min(scores)}")
    print(f"Mean steps: {np.mean(steps_history):.2f}")
    print(f"Full clears: {full_clears}/{config.evaluation.episodes}")


if __name__ == "__main__":
    main()
