from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from agent.network import RainbowNetwork
from agent.replay_buffer import NStepTransitionAccumulator, PrioritizedReplayBuffer
from agent.utils import beta_by_frame, categorical_projection, hard_update
from config import AgentConfig


class RainbowAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AgentConfig,
        device: str,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)

        self.online_net = RainbowNetwork(
            input_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            noisy_std=config.noisy_std,
        ).to(self.device)
        self.target_net = RainbowNetwork(
            input_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            noisy_std=config.noisy_std,
        ).to(self.device)

        hard_update(self.target_net, self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            state_dim=state_dim,
            alpha=config.per_alpha,
            epsilon=config.per_epsilon,
        )
        self.n_step_accumulator = NStepTransitionAccumulator(
            n_step=config.n_step,
            gamma=config.gamma,
        )
        self.support = self.online_net.support

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if evaluate:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()

        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transitions = self.n_step_accumulator.push(state, action, reward, next_state, done)
        for transition in transitions:
            transition_state, transition_action, transition_reward, transition_next_state, transition_done, steps = transition
            self.replay_buffer.add(
                state=transition_state,
                action=transition_action,
                reward=transition_reward,
                next_state=transition_next_state,
                done=transition_done,
                step_count=steps,
            )

    def learn(self, frame_idx: int) -> float | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        beta = beta_by_frame(
            frame_idx=frame_idx,
            beta_start=self.config.per_beta_start,
            beta_frames=self.config.per_beta_frames,
        )
        batch = self.replay_buffer.sample(self.config.batch_size, beta)

        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)
        step_counts = torch.as_tensor(batch["step_counts"], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)
        batch_indices = torch.arange(states.size(0), device=self.device)
        indices = batch["indices"]

        self.online_net.train()
        self.online_net.reset_noise()
        self.target_net.train()
        self.target_net.reset_noise()

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_dist = self.target_net.dist(next_states)
            next_dist = next_dist[batch_indices, next_actions]
            discounts = torch.pow(
                torch.full_like(step_counts, self.config.gamma),
                step_counts,
            )
            target_dist = categorical_projection(
                next_dist=next_dist,
                rewards=rewards,
                dones=dones,
                discounts=discounts,
                support=self.support,
                v_min=self.config.v_min,
                v_max=self.config.v_max,
                num_atoms=self.config.num_atoms,
            )

        dist = self.online_net.dist(states)
        chosen_action_dist = dist[batch_indices, actions]
        log_probabilities = torch.log(chosen_action_dist.clamp(min=1e-6))
        elementwise_loss = -(target_dist * log_probabilities).sum(dim=1)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        updated_priorities = elementwise_loss.detach().cpu().numpy() + self.config.per_epsilon
        self.replay_buffer.update_priorities(indices, updated_priorities)

        self.target_net.eval()
        return float(loss.item())

    def update_target(self) -> None:
        hard_update(self.target_net, self.online_net)
        self.target_net.eval()

    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        include_buffer: bool = False,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "agent_config": self.config.__dict__,
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        if include_buffer:
            checkpoint["replay_buffer"] = self.replay_buffer.state_dict()
        torch.save(checkpoint, path)

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        load_buffer: bool = False,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path)
        try:
            # These checkpoints are created locally by this project and can
            # contain replay-buffer arrays, so PyTorch 2.6+ must load them
            # with weights_only=False.
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["online_state_dict"])
        )
        self.support = self.online_net.support
        self.config.v_min = float(self.online_net.support[0].item())
        self.config.v_max = float(self.online_net.support[-1].item())
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_buffer and "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self.target_net.eval()
        return checkpoint.get("metadata", {})

    def warm_start_from_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        source_state = checkpoint["online_state_dict"]
        target_state = self.online_net.state_dict()
        copied_keys: list[str] = []

        for name, target_tensor in target_state.items():
            if name == "support":
                # A warm-start copies policy features, not the value support.
                # This keeps the new C51 range from config.py instead of
                # silently restoring the old checkpoint range.
                continue

            source_tensor = source_state.get(name)
            if source_tensor is None:
                continue

            if source_tensor.shape == target_tensor.shape:
                target_state[name] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
                copied_keys.append(name)
                continue

            if name == "feature_layer.0.weight" and source_tensor.ndim == 2 and target_tensor.ndim == 2:
                if source_tensor.shape[0] != target_tensor.shape[0]:
                    continue
                if source_tensor.shape[1] > target_tensor.shape[1]:
                    continue
                initialized_tensor = target_tensor.clone()
                # The hybrid grid observation starts with the rich features.
                # Extra grid-feature weights start neutral so the warm-start
                # policy initially behaves like the rich model.
                initialized_tensor.zero_()
                initialized_tensor[:, : source_tensor.shape[1]] = source_tensor.to(
                    device=target_tensor.device,
                    dtype=target_tensor.dtype,
                )
                target_state[name] = initialized_tensor
                copied_keys.append(name)

        self.online_net.load_state_dict(target_state)
        hard_update(self.target_net, self.online_net)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            state_dim=self.state_dim,
            alpha=self.config.per_alpha,
            epsilon=self.config.per_epsilon,
        )
        self.n_step_accumulator.clear()
        return {
            "source_state_dim": int(checkpoint["state_dim"]),
            "target_state_dim": self.state_dim,
            "copied_keys": copied_keys,
            "source_metadata": checkpoint.get("metadata", {}),
        }

    def set_eval_mode(self) -> None:
        self.online_net.eval()
        self.target_net.eval()


def peek_checkpoint_state_dim(path: str | Path, device: str = "cpu") -> int:
    checkpoint_path = Path(path)
    load_device = torch.device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=load_device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=load_device)
    return int(checkpoint["state_dim"])
