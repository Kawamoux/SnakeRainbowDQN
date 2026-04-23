from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class NStepTransitionAccumulator:
    def __init__(self, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be at least 1.")
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque[Transition] = deque()

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> list[tuple[np.ndarray, int, float, np.ndarray, bool, int]]:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

        ready_transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []

        if done:
            while self.buffer:
                ready_transitions.append(self._build_transition())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n_step:
            ready_transitions.append(self._build_transition())
            self.buffer.popleft()

        return ready_transitions

    def clear(self) -> None:
        self.buffer.clear()

    def _build_transition(self) -> tuple[np.ndarray, int, float, np.ndarray, bool, int]:
        reward = 0.0
        next_state = self.buffer[0].next_state
        done = False
        steps = 0

        for step_idx, transition in enumerate(self.buffer):
            reward += (self.gamma ** step_idx) * transition.reward
            next_state = transition.next_state
            done = transition.done
            steps = step_idx + 1
            if transition.done or steps >= self.n_step:
                break

        first = self.buffer[0]
        return first.state, first.action, reward, next_state, done, steps


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float,
        epsilon: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be strictly positive.")

        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.epsilon = epsilon

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.step_counts = np.ones(capacity, dtype=np.int32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        step_count: int,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.step_counts[self.position] = step_count
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict[str, Any]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        priorities = self.priorities[: self.size]
        scaled_priorities = np.power(priorities, self.alpha)
        probabilities = scaled_priorities / scaled_priorities.sum()

        replace = self.size < batch_size
        indices = np.random.choice(self.size, size=batch_size, replace=replace, p=probabilities)

        weights = np.power(self.size * probabilities[indices], -beta)
        weights /= weights.max()

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "step_counts": self.step_counts[indices],
            "weights": weights.astype(np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        safe_priorities = np.maximum(priorities, self.epsilon)
        self.priorities[indices] = safe_priorities
        self.max_priority = max(self.max_priority, float(safe_priorities.max()))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
            "states": self.states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_states": self.next_states[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
            "step_counts": self.step_counts[: self.size].copy(),
            "priorities": self.priorities[: self.size].copy(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.max_priority = float(state_dict["max_priority"])

        self.states[: self.size] = state_dict["states"]
        self.actions[: self.size] = state_dict["actions"]
        self.rewards[: self.size] = state_dict["rewards"]
        self.next_states[: self.size] = state_dict["next_states"]
        self.dones[: self.size] = state_dict["dones"]
        self.step_counts[: self.size] = state_dict["step_counts"]
        self.priorities[: self.size] = state_dict["priorities"]
