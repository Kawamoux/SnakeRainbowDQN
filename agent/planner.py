from __future__ import annotations

from collections import deque
from typing import Any


Position = tuple[int, int]


class SnakePlanner:
    """Symbolic AI controller for reliable Snake play.

    The default strategy follows a Hamiltonian cycle when possible. On an even
    grid this gives a conservative route that can eventually visit every cell
    without self-collision. A small search fallback handles the first moves if
    the initial snake is not aligned with the cycle.
    """

    DIRECTIONS: tuple[Position, ...] = (
        (1, 0),   # right
        (0, 1),   # down
        (-1, 0),  # left
        (0, -1),  # up
    )

    def __init__(self, width: int, height: int, allow_shortcuts: bool = False) -> None:
        self.width = width
        self.height = height
        self.allow_shortcuts = allow_shortcuts
        self.cycle = self._build_hamiltonian_cycle(width, height)
        self.cycle_index = (
            {position: index for index, position in enumerate(self.cycle)}
            if self.cycle is not None
            else {}
        )

    @property
    def has_cycle(self) -> bool:
        return self.cycle is not None

    def select_action(self, env: Any, preferred_action: int | None = None) -> int:
        legal_actions = self._legal_actions(env)
        if not legal_actions:
            return 0

        if self.allow_shortcuts:
            if (
                preferred_action is not None
                and preferred_action in legal_actions
                and self._action_keeps_tail_reachable(env, preferred_action)
            ):
                return int(preferred_action)

            food_action = self._safe_food_action(env)
            if food_action is not None:
                return food_action

        cycle_action = self._cycle_action(env)
        if cycle_action is not None and cycle_action in legal_actions:
            return cycle_action

        if self.cycle is None:
            food_action = self._safe_food_action(env)
            if food_action is not None:
                return food_action

        tail_action = self._tail_chasing_action(env)
        if tail_action is not None:
            return tail_action

        return self._max_space_action(env, legal_actions)

    def _cycle_action(self, env: Any) -> int | None:
        if self.cycle is None:
            return None

        head = env.snake[0]
        head_index = self.cycle_index.get(head)
        if head_index is None:
            return None

        next_cycle_cell = self.cycle[(head_index + 1) % len(self.cycle)]
        return self._action_for_target(env, next_cycle_cell)

    def _safe_food_action(self, env: Any) -> int | None:
        snake = list(env.snake)
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake)
        blocked.discard(head)
        blocked.discard(tail)

        path = self._bfs_path(head, env.food, blocked)
        if path is None or len(path) < 2:
            return None

        action = self._action_for_target(env, path[1])
        if action is None or action not in self._legal_actions(env):
            return None

        if not self._path_keeps_tail_reachable(env, path):
            return None
        return action

    def _tail_chasing_action(self, env: Any) -> int | None:
        snake = list(env.snake)
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake)
        blocked.discard(head)
        blocked.discard(tail)

        path = self._bfs_path(head, tail, blocked)
        if path is None or len(path) < 2:
            return None

        action = self._action_for_target(env, path[1])
        if action is None:
            return None
        if action not in self._legal_actions(env):
            return None
        return action

    def _max_space_action(self, env: Any, legal_actions: list[int]) -> int:
        best_action = legal_actions[0]
        best_space = -1

        for action in legal_actions:
            simulated_snake = self._simulate_action(env, action)
            if simulated_snake is None:
                continue
            space = self._reachable_space(simulated_snake)
            if space > best_space:
                best_space = space
                best_action = action

        return best_action

    def _action_for_target(self, env: Any, target: Position) -> int | None:
        for action in range(3):
            next_direction = self._turn(env.direction_idx, action)
            if self._next_position(env.snake[0], next_direction) == target:
                return action
        return None

    def _legal_actions(self, env: Any) -> list[int]:
        return [
            action
            for action in range(3)
            if self._simulate_action(env, action) is not None
        ]

    def _simulate_action(self, env: Any, action: int) -> list[Position] | None:
        snake = list(env.snake)
        new_direction = self._turn(env.direction_idx, action)
        new_head = self._next_position(snake[0], new_direction)
        will_grow = new_head == env.food

        if self._collides(new_head, snake, will_grow):
            return None

        if will_grow:
            return [new_head, *snake]
        return [new_head, *snake[:-1]]

    def _action_keeps_tail_reachable(self, env: Any, action: int) -> bool:
        simulated_snake = self._simulate_action(env, action)
        if simulated_snake is None:
            return False
        return self._tail_is_reachable(simulated_snake)

    def _path_keeps_tail_reachable(self, env: Any, path: list[Position]) -> bool:
        snake = list(env.snake)

        for next_head in path[1:]:
            will_grow = next_head == env.food
            if self._collides(next_head, snake, will_grow):
                return False
            if will_grow:
                snake = [next_head, *snake]
            else:
                snake = [next_head, *snake[:-1]]

        return self._tail_is_reachable(snake)

    def _tail_is_reachable(self, snake: list[Position]) -> bool:
        if len(snake) >= self.width * self.height:
            return True
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake)
        blocked.discard(head)
        blocked.discard(tail)
        return tail in self._flood_fill(head, blocked)

    def _reachable_space(self, snake: list[Position]) -> int:
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake)
        blocked.discard(head)
        blocked.discard(tail)
        return len(self._flood_fill(head, blocked))

    def _collides(self, position: Position, snake: list[Position], grow: bool) -> bool:
        x, y = position
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        if grow:
            return position in snake
        return position in snake[:-1]

    def _bfs_path(
        self,
        start: Position,
        target: Position,
        blocked: set[Position],
    ) -> list[Position] | None:
        if target in blocked and target != start:
            return None

        frontier: deque[Position] = deque([start])
        parent: dict[Position, Position | None] = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == target:
                return self._reconstruct_path(parent, target)

            for direction in self.DIRECTIONS:
                candidate = current[0] + direction[0], current[1] + direction[1]
                x, y = candidate
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue
                if candidate in blocked or candidate in parent:
                    continue
                parent[candidate] = current
                frontier.append(candidate)

        return None

    def _flood_fill(self, start: Position, blocked: set[Position]) -> set[Position]:
        visited = {start}
        frontier: deque[Position] = deque([start])

        while frontier:
            current = frontier.popleft()
            for direction in self.DIRECTIONS:
                candidate = current[0] + direction[0], current[1] + direction[1]
                x, y = candidate
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue
                if candidate in blocked or candidate in visited:
                    continue
                visited.add(candidate)
                frontier.append(candidate)

        return visited

    @staticmethod
    def _reconstruct_path(
        parent: dict[Position, Position | None],
        target: Position,
    ) -> list[Position]:
        path = [target]
        current = target
        while parent[current] is not None:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    @classmethod
    def _turn(cls, direction_idx: int, action: int) -> int:
        if action == 0:
            return direction_idx
        if action == 1:
            return (direction_idx - 1) % len(cls.DIRECTIONS)
        return (direction_idx + 1) % len(cls.DIRECTIONS)

    @classmethod
    def _next_position(cls, position: Position, direction_idx: int) -> Position:
        delta_x, delta_y = cls.DIRECTIONS[direction_idx]
        return position[0] + delta_x, position[1] + delta_y

    @classmethod
    def _build_hamiltonian_cycle(cls, width: int, height: int) -> list[Position] | None:
        if width < 2 or height < 2:
            return None
        if height % 2 == 0:
            return cls._horizontal_hamiltonian_cycle(width, height)
        if width % 2 == 0:
            rotated = cls._horizontal_hamiltonian_cycle(height, width)
            return [(y, x) for x, y in rotated]
        return None

    @staticmethod
    def _horizontal_hamiltonian_cycle(width: int, height: int) -> list[Position]:
        cycle: list[Position] = []

        for y in range(height):
            if y == 0:
                cycle.extend((x, y) for x in range(width))
            elif y % 2 == 1:
                cycle.extend((x, y) for x in range(width - 1, 0, -1))
            else:
                cycle.extend((x, y) for x in range(1, width))

        cycle.extend((0, y) for y in range(height - 1, 0, -1))
        return cycle
