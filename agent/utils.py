from __future__ import annotations

import torch


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def beta_by_frame(frame_idx: int, beta_start: float, beta_frames: int) -> float:
    progress = min(frame_idx / max(beta_frames, 1), 1.0)
    return beta_start + progress * (1.0 - beta_start)


def categorical_projection(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discounts: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
    num_atoms: int,
) -> torch.Tensor:
    batch_size = rewards.size(0)
    delta_z = (v_max - v_min) / (num_atoms - 1)

    target_support = rewards + (1.0 - dones) * discounts * support.view(1, -1)
    target_support = target_support.clamp(v_min, v_max)

    b = (target_support - v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()

    projected_dist = torch.zeros_like(next_dist)
    offset = (
        torch.arange(batch_size, device=next_dist.device)
        .unsqueeze(1)
        .expand(batch_size, num_atoms)
        * num_atoms
    )

    lower_flat = (lower + offset).view(-1)
    upper_flat = (upper + offset).view(-1)
    projected_flat = projected_dist.view(-1)

    same_index = lower == upper
    if same_index.any():
        projected_flat.index_add_(
            0,
            lower_flat[same_index.view(-1)],
            next_dist[same_index].view(-1),
        )

    different_index = ~same_index
    if different_index.any():
        projected_flat.index_add_(
            0,
            lower_flat[different_index.view(-1)],
            (next_dist * (upper.float() - b))[different_index].view(-1),
        )
        projected_flat.index_add_(
            0,
            upper_flat[different_index.view(-1)],
            (next_dist * (b - lower.float()))[different_index].view(-1),
        )

    return projected_dist
