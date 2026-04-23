from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from config import EnvConfig


Position = tuple[int, int]


class SnakeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    _DIRECTIONS: tuple[Position, ...] = (
        (1, 0),   # right
        (0, 1),   # down
        (-1, 0),  # left
        (0, -1),  # up
    )

    def __init__(self, config: EnvConfig, render_mode: str | None = None) -> None:
        super().__init__()
        if config.grid_width < 5 or config.grid_height < 5:
            raise ValueError("The Snake grid must be at least 5x5.")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if config.observation_mode not in {
            "compact",
            "rich",
            "grid",
            "grid_hybrid_legacy",
            "grid_legacy",
        }:
            raise ValueError(f"Unsupported observation mode: {config.observation_mode}")

        self.config = config
        self.render_mode = render_mode
        self.metadata["render_fps"] = config.render_fps

        self.action_space = spaces.Discrete(config.action_size)
        observation_low = -1.0 if config.observation_mode in {"rich", "grid", "grid_hybrid_legacy"} else 0.0
        self.observation_space = spaces.Box(
            low=observation_low,
            high=1.0,
            shape=(config.observation_size,),
            dtype=np.float32,
        )

        self.window_size = (
            self.config.grid_width * self.config.cell_size,
            self.config.grid_height * self.config.cell_size,
        )

        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None

        self.snake: deque[Position] = deque()
        self.snake_set: set[Position] = set()
        self.direction_idx = 0
        self.food: Position = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.was_trapped = False
        self.was_tail_unreachable = False
        self.was_low_space = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        center_x = self.config.grid_width // 2
        center_y = self.config.grid_height // 2
        self.direction_idx = int(self.np_random.integers(0, len(self._DIRECTIONS)))

        head = (center_x, center_y)
        direction = self._DIRECTIONS[self.direction_idx]
        body_1 = (head[0] - direction[0], head[1] - direction[1])
        body_2 = (body_1[0] - direction[0], body_1[1] - direction[1])

        self.snake = deque([head, body_1, body_2])
        self.snake_set = set(self.snake)
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.was_trapped = False
        self.was_tail_unreachable = False
        self.was_low_space = False

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        self.steps_since_food += 1
        reward_breakdown = self._empty_reward_breakdown()

        previous_distance = self._manhattan_distance(self.snake[0], self.food)
        new_direction_idx = self._turn(self.direction_idx, action)
        new_head = self._next_position(self.snake[0], new_direction_idx)
        will_grow = new_head == self.food

        terminated = False
        truncated = False
        trap_detected = False

        if self._is_collision(new_head, grow=will_grow):
            reward_breakdown["death"] = self.config.death_penalty
            reward = reward_breakdown["death"]
            terminated = True
            self.was_trapped = False
            info = self._get_info(collision=True, reward_breakdown=reward_breakdown)
            return self._get_observation(), reward, terminated, truncated, info

        if will_grow:
            self.direction_idx = new_direction_idx
            self.snake.appendleft(new_head)
            self.snake_set.add(new_head)
            self.score += 1
            self.steps_since_food = 0
            reward_breakdown["food"] = self.config.food_reward
            reward = reward_breakdown["food"]
            if len(self.snake) == self.config.grid_width * self.config.grid_height:
                terminated = True
                reward_breakdown["full_clear"] = self.config.full_clear_reward
                reward += reward_breakdown["full_clear"]
            else:
                self.food = self._place_food()
        else:
            tail = self.snake.pop()
            if tail != new_head:
                self.snake_set.remove(tail)
            self.direction_idx = new_direction_idx
            self.snake.appendleft(new_head)
            self.snake_set.add(new_head)
            reward_breakdown["step"] = self.config.step_penalty
            reward = reward_breakdown["step"]
            new_distance = self._manhattan_distance(self.snake[0], self.food)
            if new_distance < previous_distance:
                reward_breakdown["distance"] = self.config.closer_reward
                reward += reward_breakdown["distance"]
            elif new_distance > previous_distance:
                reward_breakdown["distance"] = self.config.farther_penalty
                reward += reward_breakdown["distance"]

        trap_detected = self._is_likely_trapped()
        if trap_detected and not self.was_trapped:
            reward_breakdown["trap"] = self.config.trap_penalty
            reward += reward_breakdown["trap"]
        self.was_trapped = trap_detected

        if not terminated and self.config.safety_shaping:
            safety_reward = self._safety_shaping_reward()
            if safety_reward:
                reward_breakdown["safety"] = safety_reward
                reward += safety_reward

        if not terminated and self.steps_since_food >= self.config.max_steps_without_food:
            truncated = True
            reward_breakdown["timeout"] = self.config.timeout_penalty
            reward += reward_breakdown["timeout"]

        observation = self._get_observation()
        info = self._get_info(
            ate_food=will_grow,
            trap_detected=trap_detected,
            reward_breakdown=reward_breakdown,
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        self._initialize_pygame()
        assert self.window is not None

        canvas = pygame.Surface(self.window_size)
        canvas.fill((18, 22, 29))
        self._draw_grid(canvas)
        self._draw_food(canvas)
        self._draw_snake(canvas)
        self._draw_score(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            assert self.clock is not None
            self.clock.tick(self.config.render_fps)
            return None

        frame = pygame.surfarray.array3d(canvas)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None

    def _get_observation(self) -> np.ndarray:
        if self.config.observation_mode == "compact":
            return self._get_compact_observation()
        if self.config.observation_mode == "grid":
            return self._get_grid_observation()
        if self.config.observation_mode == "grid_hybrid_legacy":
            return self._get_grid_hybrid_legacy_observation()
        if self.config.observation_mode == "grid_legacy":
            return self._get_grid_only_observation()
        return self._get_rich_observation()

    def _get_compact_observation(self) -> np.ndarray:
        head = self.snake[0]
        left_direction_idx = (self.direction_idx - 1) % len(self._DIRECTIONS)
        right_direction_idx = (self.direction_idx + 1) % len(self._DIRECTIONS)

        danger_straight = float(
            self._is_collision(self._next_position(head, self.direction_idx), grow=False)
        )
        danger_left = float(
            self._is_collision(self._next_position(head, left_direction_idx), grow=False)
        )
        danger_right = float(
            self._is_collision(self._next_position(head, right_direction_idx), grow=False)
        )

        dir_right = float(self.direction_idx == 0)
        dir_down = float(self.direction_idx == 1)
        dir_left = float(self.direction_idx == 2)
        dir_up = float(self.direction_idx == 3)

        food_left = float(self.food[0] < head[0])
        food_right = float(self.food[0] > head[0])
        food_up = float(self.food[1] < head[1])
        food_down = float(self.food[1] > head[1])

        return np.array(
            [
                danger_straight,
                danger_left,
                danger_right,
                dir_up,
                dir_down,
                dir_left,
                dir_right,
                food_up,
                food_down,
                food_left,
                food_right,
            ],
            dtype=np.float32,
        )

    def _get_rich_observation(self) -> np.ndarray:
        compact = self._get_compact_observation()
        head = self.snake[0]
        grid_extent = float(max(self.config.grid_width - 1, self.config.grid_height - 1, 1))
        total_cells = float(self.config.grid_width * self.config.grid_height)
        free_ratio_scale = max(total_cells, 1.0)

        food_dx = (self.food[0] - head[0]) / max(self.config.grid_width - 1, 1)
        food_dy = (self.food[1] - head[1]) / max(self.config.grid_height - 1, 1)

        relative_direction_indices = (
            self.direction_idx,
            (self.direction_idx - 1) % len(self._DIRECTIONS),
            (self.direction_idx + 1) % len(self._DIRECTIONS),
        )
        wall_distances = [
            self._distance_to_wall(head, direction_idx) / grid_extent
            for direction_idx in relative_direction_indices
        ]
        body_distances = [
            self._distance_to_body(head, direction_idx) / grid_extent
            for direction_idx in relative_direction_indices
        ]

        reachable_ratios: list[float] = []
        tail_reachable_flags: list[float] = []
        eat_flags: list[float] = []
        for action in range(self.action_space.n):
            action_metrics = self._simulate_action_metrics(action)
            reachable_ratios.append(action_metrics["reachable_ratio"])
            tail_reachable_flags.append(float(action_metrics["tail_reachable"]))
            eat_flags.append(float(action_metrics["will_grow"]))

        length_ratio = len(self.snake) / total_cells
        steps_since_food_ratio = self.steps_since_food / max(self.config.max_steps_without_food, 1)

        return np.array(
            [
                *compact.tolist(),
                float(food_dx),
                float(food_dy),
                *wall_distances,
                *body_distances,
                *reachable_ratios,
                *tail_reachable_flags,
                *eat_flags,
                float(length_ratio),
                float(steps_since_food_ratio),
            ],
            dtype=np.float32,
        )

    def _get_grid_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self._get_grid_hybrid_legacy_observation(),
                self._get_food_path_observation(),
            ]
        ).astype(np.float32)

    def _get_grid_hybrid_legacy_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self._get_rich_observation(),
                self._get_grid_only_observation(),
            ]
        ).astype(np.float32)

    def _get_food_path_observation(self) -> np.ndarray:
        reachable_flags: list[float] = []
        distance_ratios: list[float] = []
        max_distance = max(float(self.config.grid_width * self.config.grid_height), 1.0)

        for action in range(self.action_space.n):
            new_direction_idx = self._turn(self.direction_idx, action)
            new_head = self._next_position(self.snake[0], new_direction_idx)
            will_grow = new_head == self.food
            if self._is_collision(new_head, grow=will_grow):
                reachable_flags.append(0.0)
                distance_ratios.append(1.0)
                continue
            if will_grow:
                reachable_flags.append(1.0)
                distance_ratios.append(0.0)
                continue

            blocked = set(self.snake_set)
            blocked.discard(self.snake[-1])
            blocked.discard(new_head)
            distance = self._shortest_path_distance(new_head, self.food, blocked)
            reachable_flags.append(float(distance is not None))
            distance_ratios.append(float(distance / max_distance) if distance is not None else 1.0)

        return np.array([*reachable_flags, *distance_ratios], dtype=np.float32)

    def _get_grid_only_observation(self) -> np.ndarray:
        total_cells = float(self.config.grid_width * self.config.grid_height)
        head_plane = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.float32)
        body_order_plane = np.zeros_like(head_plane)
        food_plane = np.zeros_like(head_plane)

        snake_length = max(len(self.snake), 1)
        for index, (x, y) in enumerate(self.snake):
            # Full body order matters in the endgame; occupancy alone aliases too much.
            body_order_plane[y, x] = (snake_length - index) / snake_length

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        head_plane[head_y, head_x] = 1.0
        food_plane[food_y, food_x] = 1.0

        direction = np.array(
            [
                float(self.direction_idx == 0),
                float(self.direction_idx == 1),
                float(self.direction_idx == 2),
                float(self.direction_idx == 3),
            ],
            dtype=np.float32,
        )
        extras = np.array(
            [
                len(self.snake) / total_cells,
                self.steps_since_food / max(self.config.max_steps_without_food, 1),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                head_plane.reshape(-1),
                body_order_plane.reshape(-1),
                food_plane.reshape(-1),
                direction,
                extras,
            ]
        ).astype(np.float32)

    def _get_info(
        self,
        ate_food: bool = False,
        collision: bool = False,
        trap_detected: bool = False,
        reward_breakdown: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "score": self.score,
            "length": len(self.snake),
            "steps": self.steps,
            "ate_food": ate_food,
            "collision": collision,
            "trap_detected": trap_detected,
            "reward_breakdown": reward_breakdown or self._empty_reward_breakdown(),
        }

    @staticmethod
    def _empty_reward_breakdown() -> dict[str, float]:
        return {
            "food": 0.0,
            "death": 0.0,
            "step": 0.0,
            "distance": 0.0,
            "full_clear": 0.0,
            "trap": 0.0,
            "safety": 0.0,
            "timeout": 0.0,
        }

    def _place_food(self) -> Position:
        free_cells = [
            (x, y)
            for x in range(self.config.grid_width)
            for y in range(self.config.grid_height)
            if (x, y) not in self.snake_set
        ]
        if not free_cells:
            return self.snake[0]
        index = int(self.np_random.integers(0, len(free_cells)))
        return free_cells[index]

    def _turn(self, direction_idx: int, action: int) -> int:
        if action == 0:
            return direction_idx
        if action == 1:
            return (direction_idx - 1) % len(self._DIRECTIONS)
        return (direction_idx + 1) % len(self._DIRECTIONS)

    def _next_position(self, position: Position, direction_idx: int) -> Position:
        delta_x, delta_y = self._DIRECTIONS[direction_idx]
        return position[0] + delta_x, position[1] + delta_y

    def _is_collision(self, position: Position, grow: bool) -> bool:
        x, y = position
        if x < 0 or x >= self.config.grid_width or y < 0 or y >= self.config.grid_height:
            return True

        if grow:
            return position in self.snake_set

        tail = self.snake[-1]
        return position in self.snake_set and position != tail

    def _distance_to_wall(self, position: Position, direction_idx: int) -> float:
        x, y = position
        if direction_idx == 0:
            return float(self.config.grid_width - 1 - x)
        if direction_idx == 1:
            return float(self.config.grid_height - 1 - y)
        if direction_idx == 2:
            return float(x)
        return float(y)

    def _distance_to_body(self, position: Position, direction_idx: int) -> float:
        distance = 0
        current_position = position
        while True:
            current_position = self._next_position(current_position, direction_idx)
            x, y = current_position
            if x < 0 or x >= self.config.grid_width or y < 0 or y >= self.config.grid_height:
                break
            distance += 1
            if current_position in self.snake_set:
                return float(distance)
        return self._distance_to_wall(position, direction_idx)

    def _simulate_action_metrics(self, action: int) -> dict[str, float | bool]:
        new_direction_idx = self._turn(self.direction_idx, action)
        new_head = self._next_position(self.snake[0], new_direction_idx)
        will_grow = new_head == self.food
        collision = self._is_collision(new_head, grow=will_grow)
        if collision:
            return {
                "collision": True,
                "will_grow": will_grow,
                "reachable_ratio": 0.0,
                "tail_reachable": False,
            }

        simulated_snake = list(self.snake)
        if not will_grow and simulated_snake:
            simulated_snake = simulated_snake[:-1]
        simulated_snake = [new_head, *simulated_snake]
        simulated_tail = simulated_snake[-1]
        blocked_for_space = set(simulated_snake)
        blocked_for_space.discard(new_head)
        if not will_grow:
            blocked_for_space.discard(simulated_tail)

        blocked_for_tail = set(simulated_snake)
        blocked_for_tail.discard(new_head)
        blocked_for_tail.discard(simulated_tail)

        reachable_cells = self._flood_fill_count(new_head, blocked_for_space)
        reachable_ratio = reachable_cells / max(float(self.config.grid_width * self.config.grid_height), 1.0)
        tail_reachable = simulated_tail in self._flood_fill_positions(new_head, blocked_for_tail)
        return {
            "collision": False,
            "will_grow": will_grow,
            "reachable_ratio": float(reachable_ratio),
            "tail_reachable": bool(tail_reachable),
        }

    def _is_likely_trapped(self) -> bool:
        reachable_space = self._reachable_space_from_head()
        minimum_safe_space = len(self.snake) + self.config.trap_min_open_space_buffer
        return reachable_space < minimum_safe_space

    def _safety_shaping_reward(self) -> float:
        if len(self.snake) < self.config.safety_min_length:
            self.was_tail_unreachable = False
            self.was_low_space = False
            return 0.0

        reward = 0.0
        reachable_space = self._reachable_space_from_head()
        minimum_safe_space = len(self.snake) + self.config.trap_min_open_space_buffer
        low_space = reachable_space < minimum_safe_space
        if low_space and not self.was_low_space:
            missing_space = minimum_safe_space - reachable_space
            reward += max(
                self.config.low_space_penalty_max,
                -self.config.low_space_penalty_scale * missing_space,
            )
        self.was_low_space = low_space

        tail_unreachable = not self._is_tail_reachable_from_head()
        if tail_unreachable and not self.was_tail_unreachable:
            reward += self.config.tail_unreachable_penalty
        self.was_tail_unreachable = tail_unreachable

        return float(reward)

    def _is_tail_reachable_from_head(self) -> bool:
        if not self.snake:
            return True
        head = self.snake[0]
        tail = self.snake[-1]
        blocked = set(self.snake_set)
        blocked.discard(head)
        blocked.discard(tail)
        return tail in self._flood_fill_positions(head, blocked)

    def _reachable_space_from_head(self) -> int:
        head = self.snake[0]
        blocked = set(self.snake_set)
        blocked.discard(head)
        if self.snake:
            blocked.discard(self.snake[-1])
        return self._flood_fill_count(head, blocked)

    def _flood_fill_count(self, start: Position, blocked: set[Position]) -> int:
        return len(self._flood_fill_positions(start, blocked))

    def _flood_fill_positions(self, start: Position, blocked: set[Position]) -> set[Position]:
        visited = {start}
        frontier: deque[Position] = deque([start])

        while frontier:
            position = frontier.popleft()
            for direction_idx in range(len(self._DIRECTIONS)):
                next_position = self._next_position(position, direction_idx)
                x, y = next_position
                if x < 0 or x >= self.config.grid_width or y < 0 or y >= self.config.grid_height:
                    continue
                if next_position in blocked or next_position in visited:
                    continue
                visited.add(next_position)
                frontier.append(next_position)

        return visited

    def _shortest_path_distance(
        self,
        start: Position,
        target: Position,
        blocked: set[Position],
    ) -> int | None:
        if start == target:
            return 0
        visited = {start}
        frontier: deque[tuple[Position, int]] = deque([(start, 0)])

        while frontier:
            position, distance = frontier.popleft()
            for direction_idx in range(len(self._DIRECTIONS)):
                next_position = self._next_position(position, direction_idx)
                x, y = next_position
                if x < 0 or x >= self.config.grid_width or y < 0 or y >= self.config.grid_height:
                    continue
                if next_position in blocked or next_position in visited:
                    continue
                if next_position == target:
                    return distance + 1
                visited.add(next_position)
                frontier.append((next_position, distance + 1))

        return None

    @staticmethod
    def _manhattan_distance(first: Position, second: Position) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _initialize_pygame(self) -> None:
        if self.window is not None:
            return

        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Snake Rainbow DQN")
        else:
            self.window = pygame.Surface(self.window_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)

    def _draw_grid(self, canvas: pygame.Surface) -> None:
        line_color = (31, 38, 50)
        for x in range(0, self.window_size[0], self.config.cell_size):
            pygame.draw.line(canvas, line_color, (x, 0), (x, self.window_size[1]))
        for y in range(0, self.window_size[1], self.config.cell_size):
            pygame.draw.line(canvas, line_color, (0, y), (self.window_size[0], y))

    def _draw_food(self, canvas: pygame.Surface) -> None:
        rect = pygame.Rect(
            self.food[0] * self.config.cell_size,
            self.food[1] * self.config.cell_size,
            self.config.cell_size,
            self.config.cell_size,
        )
        pygame.draw.rect(canvas, (220, 76, 70), rect.inflate(-6, -6), border_radius=6)

    def _draw_snake(self, canvas: pygame.Surface) -> None:
        for index, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[0] * self.config.cell_size,
                segment[1] * self.config.cell_size,
                self.config.cell_size,
                self.config.cell_size,
            )
            color = (95, 214, 110) if index == 0 else (56, 166, 92)
            pygame.draw.rect(canvas, color, rect.inflate(-4, -4), border_radius=6)

    def _draw_score(self, canvas: pygame.Surface) -> None:
        if self.font is None:
            return
        text = self.font.render(f"Score: {self.score}", True, (230, 235, 241))
        canvas.blit(text, (8, 8))
