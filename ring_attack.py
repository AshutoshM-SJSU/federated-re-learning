"""Reusable RING-style collusion attack utilities for federated learning.

This module isolates the collusion logic from the experiment runner found in the
reference implementation. It is model- and dataset-agnostic: it operates on
PyTorch state_dict mappings and therefore works with CNNs, ResNets, LSTMs, and
MLPs as long as the caller supplies attacker model states and the current global
model state.

Scope
-----
This file handles the post-training RING coordination step:

1. attackers train normally/maliciously and produce target local model states;
2. the expected DP noise scale is computed for each attacker;
3. colluders receive zero-sum noise, so their individual submissions appear
   noisy while the colluding group's mean noise cancels;
4. crafted model states and corresponding model updates are returned.

It does NOT replace local DP-SGD. Per-example gradient clipping/noising during
local optimization should remain in the client trainer. Shared DP configuration
and reference noise accounting live in ``dp.py`` and are imported here so RING
uses the same privacy configuration as the rest of the FL project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from dp import (
    DPConfig,
    compute_client_noise_stds,
    compute_reference_noise_multiplier,
)

TensorDict = Mapping[str, torch.Tensor]
MutableTensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class RINGConfig:
    """Configuration for colluding attacker coordination."""

    collusion_group_size: int = -1
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def normalized_group_size(self, num_attackers: int) -> int:
        if num_attackers <= 0:
            raise ValueError("num_attackers must be positive")

        if (
            self.collusion_group_size == -1
            or self.collusion_group_size >= num_attackers
        ):
            return num_attackers

        if self.collusion_group_size <= 0:
            raise ValueError(
                "collusion_group_size must be -1 or a positive integer"
            )

        return self.collusion_group_size


def _as_per_attacker_tensors(
    target_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    num_attackers: int,
) -> List[torch.Tensor]:
    """Normalize a tensor or tensor sequence to one tensor per attacker."""

    if isinstance(target_tensors, torch.Tensor):
        return [target_tensors] * num_attackers

    tensors = list(target_tensors)

    if len(tensors) != num_attackers:
        raise ValueError(
            f"expected {num_attackers} target tensors, received {len(tensors)}"
        )

    return tensors


def _as_per_attacker_stds(
    noise_std: Union[float, int, Sequence[float]],
    num_attackers: int,
) -> List[float]:
    """Normalize scalar or sequence noise values to one std per attacker."""

    if isinstance(noise_std, (float, int, np.floating, np.integer)):
        stds = [float(noise_std)] * num_attackers
    else:
        stds = [float(x) for x in noise_std]

        if len(stds) != num_attackers:
            raise ValueError(
                f"expected {num_attackers} std values, received {len(stds)}"
            )

    if any(std < 0 for std in stds):
        raise ValueError("noise standard deviations must be non-negative")

    return stds


def generate_ring_tensors(
    target_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    *,
    num_attackers: int,
    noise_std: Union[float, int, Sequence[float]],
    collusion_group_size: int = -1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Apply RING zero-sum noise coordination to one floating-point tensor.

    Each collusion group samples independent Gaussian noise and subtracts the
    group's mean noise. Therefore, for a group containing at least two attackers,
    the added noise sums to zero up to floating-point precision.

    A one-attacker leftover group cannot cancel with another attacker and
    therefore receives ordinary Gaussian noise, matching the reference behavior.
    """

    if num_attackers <= 0:
        raise ValueError("num_attackers must be positive")

    if (clip_min is None) ^ (clip_max is None):
        raise ValueError(
            "clip_min and clip_max must either both be set or both be None"
        )

    if clip_min is not None and clip_min > clip_max:
        raise ValueError("clip_min cannot exceed clip_max")

    targets = _as_per_attacker_tensors(target_tensors, num_attackers)
    stds = _as_per_attacker_stds(noise_std, num_attackers)

    if collusion_group_size == -1 or collusion_group_size >= num_attackers:
        group_size = num_attackers
    elif collusion_group_size <= 0:
        raise ValueError("collusion_group_size must be -1 or positive")
    else:
        group_size = collusion_group_size

    crafted: List[torch.Tensor] = []
    idx = 0

    while idx < num_attackers:
        remaining = num_attackers - idx
        current_size = min(group_size, remaining)
        group_targets = targets[idx : idx + current_size]

        group_noises = [
            torch.normal(
                mean=0.0,
                std=stds[idx + j],
                size=target.shape,
                generator=generator,
                device=target.device,
                dtype=target.dtype,
            )
            for j, target in enumerate(group_targets)
        ]

        if current_size > 1:
            noise_mean = torch.stack(group_noises, dim=0).mean(dim=0)
            group_noises = [noise - noise_mean for noise in group_noises]

        for target, noise in zip(group_targets, group_noises):
            out = target + noise

            if clip_min is not None:
                out = torch.clamp(
                    out,
                    min=clip_min,
                    max=clip_max,
                )

            crafted.append(out)

        idx += current_size

    return crafted


def _validate_state_dicts(
    attacker_states: Sequence[TensorDict],
    global_state: TensorDict,
) -> None:
    """Validate attacker state_dict structure against the global model state."""

    if not attacker_states:
        raise ValueError("attacker_states cannot be empty")

    global_keys = list(global_state.keys())

    for attacker_idx, state in enumerate(attacker_states):
        if list(state.keys()) != global_keys:
            raise ValueError(
                f"attacker state {attacker_idx} does not have the same "
                "ordered keys as global_state"
            )

        for key in global_keys:
            if state[key].shape != global_state[key].shape:
                raise ValueError(
                    f"shape mismatch for parameter {key!r} in attacker state "
                    f"{attacker_idx}: {tuple(state[key].shape)} != "
                    f"{tuple(global_state[key].shape)}"
                )


def craft_ring_attack(
    *,
    attacker_states: Sequence[TensorDict],
    global_state: TensorDict,
    client_sample_counts: Sequence[int],
    learning_rate: float,
    dp: DPConfig,
    ring: RINGConfig = RINGConfig(),
    base_noise_multiplier: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[
    List[MutableTensorDict],
    List[MutableTensorDict],
    Dict[str, object],
]:
    """Craft full RING attacker submissions for one federated round.

    Parameters
    ----------
    attacker_states:
        Malicious local model states after local training. These are the
        attackers' target submissions before RING noise coordination.
    global_state:
        Current global model state used to convert crafted model states into
        model updates.
    client_sample_counts:
        Number of local samples for each attacker, in the same order as
        ``attacker_states``.
    learning_rate:
        Effective local learning rate used by the malicious clients.
    dp:
        Shared DP configuration imported from ``dp.py``.
    ring:
        RING collusion-group and optional clipping configuration.
    base_noise_multiplier:
        Optional precomputed reference noise multiplier.
    generator:
        Optional PyTorch random generator for deterministic noise generation.

    Returns
    -------
    crafted_states:
        Coordinated model states to submit as malicious clients.
    crafted_updates:
        ``crafted_state - global_state`` for each malicious client.
    metadata:
        Noise multiplier, per-attacker standard deviations, and effective
        collusion group size.
    """

    attacker_states = list(attacker_states)
    num_attackers = len(attacker_states)

    _validate_state_dicts(attacker_states, global_state)

    if len(client_sample_counts) != num_attackers:
        raise ValueError(
            "client_sample_counts must contain one entry per attacker "
            f"({num_attackers} expected, {len(client_sample_counts)} received)"
        )

    dp.validate()
    group_size = ring.normalized_group_size(num_attackers)

    if dp.enabled:
        multiplier = (
            compute_reference_noise_multiplier(dp)
            if base_noise_multiplier is None
            else float(base_noise_multiplier)
        )

        if multiplier < 0:
            raise ValueError("base_noise_multiplier must be non-negative")
    else:
        multiplier = 0.0

    stds = compute_client_noise_stds(
        client_sample_counts=client_sample_counts,
        learning_rate=learning_rate,
        dp=dp,
        base_noise_multiplier=multiplier,
    )

    crafted_states: List[MutableTensorDict] = [
        dict() for _ in range(num_attackers)
    ]
    crafted_updates: List[MutableTensorDict] = [
        dict() for _ in range(num_attackers)
    ]

    for key in global_state.keys():
        targets = [state[key] for state in attacker_states]
        global_tensor = global_state[key]

        # Some state_dict entries can be integer-valued buffers, for example
        # counters maintained by normalization layers. Gaussian noise should
        # only be applied to floating-point tensors.
        if not torch.is_floating_point(global_tensor):
            for i, target in enumerate(targets):
                crafted_states[i][key] = target.clone()
                crafted_updates[i][key] = torch.zeros_like(target)
            continue

        crafted_tensors = generate_ring_tensors(
            targets,
            num_attackers=num_attackers,
            noise_std=stds,
            collusion_group_size=group_size,
            clip_min=ring.clip_min,
            clip_max=ring.clip_max,
            generator=generator,
        )

        for i, crafted_tensor in enumerate(crafted_tensors):
            reference = global_tensor.to(
                device=crafted_tensor.device,
                dtype=crafted_tensor.dtype,
            )

            crafted_states[i][key] = crafted_tensor
            crafted_updates[i][key] = crafted_tensor - reference

    metadata: Dict[str, object] = {
        "num_attackers": num_attackers,
        "collusion_group_size": group_size,
        "dp_enabled": dp.enabled,
        "base_noise_multiplier": multiplier,
        "noise_stds": stds,
    }

    return crafted_states, crafted_updates, metadata


__all__ = [
    "RINGConfig",
    "generate_ring_tensors",
    "craft_ring_attack",
]
