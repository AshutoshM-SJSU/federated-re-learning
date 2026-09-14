"""Shared differential-privacy utilities for federated-learning experiments.

This module owns the DP configuration and the noise-accounting calculation used
by the reference RING implementation.  It is intentionally model- and
 dataset-agnostic so the same configuration can be reused by CIFAR-10, FEMNIST,
Shakespeare, and N-BaIoT experiments.

Local DP-SGD still belongs inside the client-training code because per-example
gradient clipping and noising must occur while gradients are available.  This
module provides the shared privacy parameters and accounting utilities that the
client trainer and attacks such as RING can both import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import math


@dataclass(frozen=True)
class DPConfig:
    """Differential-privacy configuration shared across the FL project.

    Parameters
    ----------
    enabled:
        If False, DP accounting/noise is disabled.
    epsilon:
        Target privacy budget epsilon.
    delta:
        Target privacy parameter delta.
    clip:
        Clipping norm used by the DP mechanism.
    local_epochs:
        Number of local training epochs per selected client.
    total_fl_epochs:
        Number of global FL rounds/epochs used by the reference accountant.
    client_fraction:
        Fraction of clients selected per global round.
    accountant_tolerance:
        Numerical tolerance passed to TensorFlow Privacy's reference accountant.
    """

    enabled: bool = True
    epsilon: float = 8.0
    delta: float = 1e-5
    clip: float = 1.0
    local_epochs: int = 1
    total_fl_epochs: int = 1
    client_fraction: float = 1.0
    accountant_tolerance: float = 1e-5

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.clip <= 0:
            raise ValueError("clip must be > 0")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be > 0")
        if self.total_fl_epochs <= 0:
            raise ValueError("total_fl_epochs must be > 0")
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.accountant_tolerance <= 0:
            raise ValueError("accountant_tolerance must be > 0")

    @property
    def accountant_epochs(self) -> float:
        """Reference accounting horizon used by the original implementation."""
        return self.total_fl_epochs * self.client_fraction * self.local_epochs


def compute_reference_noise_multiplier(dp: DPConfig) -> float:
    """Compute the base noise multiplier used by the reference implementation.

    This intentionally reproduces the supplied RING code, which calls
    ``tensorflow_privacy.compute_noise_from_budget_lib.compute_noise`` with
    dataset size and batch size both set to one.

    This function is accounting support for reproducing that implementation;
    it is not, by itself, a complete local DP-SGD training procedure.
    """

    dp.validate()
    if not dp.enabled:
        return 0.0

    try:
        from tensorflow_privacy.compute_noise_from_budget_lib import compute_noise
    except ImportError as exc:
        raise ImportError(
            "DP noise accounting requires tensorflow-privacy because the "
            "reference RING implementation uses compute_noise_from_budget_lib."
        ) from exc

    return float(
        compute_noise(
            1,
            1,
            dp.epsilon,
            dp.accountant_epochs,
            dp.delta,
            dp.accountant_tolerance,
        )
    )


def compute_client_noise_stds(
    *,
    client_sample_counts: Sequence[int],
    learning_rate: float,
    dp: DPConfig,
    base_noise_multiplier: Optional[float] = None,
) -> List[float]:
    """Compute per-client noise standard deviations used by RING.

    For client ``i`` with ``n_i`` local samples, the supplied implementation
    computes::

        sigma_i = lr * clip * sqrt(local_epochs) / n_i * noise_multiplier

    Keeping this calculation here gives every experiment one canonical DP
    accounting implementation and lets RING import it directly.
    """

    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if not client_sample_counts:
        raise ValueError("client_sample_counts cannot be empty")
    if any(n <= 0 for n in client_sample_counts):
        raise ValueError("every client sample count must be positive")

    dp.validate()
    if not dp.enabled:
        return [0.0 for _ in client_sample_counts]

    multiplier = (
        compute_reference_noise_multiplier(dp)
        if base_noise_multiplier is None
        else float(base_noise_multiplier)
    )
    if multiplier < 0:
        raise ValueError("base_noise_multiplier must be non-negative")

    return [
        learning_rate
        * dp.clip
        * math.sqrt(dp.local_epochs)
        / int(num_samples)
        * multiplier
        for num_samples in client_sample_counts
    ]


__all__ = [
    "DPConfig",
    "compute_reference_noise_multiplier",
    "compute_client_noise_stds",
]
