"""Shared differential-privacy utilities for federated-learning experiments.

This module owns the DP configuration and the reference noise-accounting
calculation used by the RING implementation.

It intentionally depends only on Google's lightweight ``dp-accounting`` package,
not the full ``tensorflow-privacy`` package. This is useful for modern PyTorch
environments, including Python 3.12.

The accounting calculation reproduces the logic used by TensorFlow Privacy's
``compute_noise_from_budget_lib.compute_noise``.

Local DP-SGD still belongs inside client training because per-example gradient
clipping and gradient noising must occur while gradients are available. This
module provides only shared privacy configuration and reference accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import math


@dataclass(frozen=True)
class DPConfig:
    """Differential-privacy configuration shared across the FL project."""

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
        """Reference accounting horizon used by the RING implementation."""
        return (
            self.total_fl_epochs
            * self.client_fraction
            * self.local_epochs
        )


def _compute_noise_from_budget(
    n: int,
    batch_size: int,
    target_epsilon: float,
    epochs: float,
    delta: float,
    noise_lower_bound: float,
) -> float:
    """Reproduce TensorFlow Privacy's compute_noise helper with dp-accounting.

    This mirrors the algorithm in
    tensorflow_privacy/privacy/analysis/compute_noise_from_budget_lib.py
    without importing TensorFlow Privacy itself.
    """

    try:
        import dp_accounting
    except ImportError as exc:
        raise ImportError(
            "DP noise accounting requires the lightweight 'dp-accounting' "
            "package. Install it with: pip install dp-accounting"
        ) from exc

    if n <= 0:
        raise ValueError("n must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > n:
        raise ValueError("n must be greater than or equal to batch_size")
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if noise_lower_bound <= 0:
        raise ValueError("noise_lower_bound must be positive")

    sampling_probability = batch_size / n

    orders = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )

    steps = int(math.ceil(epochs * n / batch_size))

    def make_event_from_noise(noise_multiplier: float):
        return dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(
                sampling_probability,
                dp_accounting.GaussianDpEvent(noise_multiplier),
            ),
            steps,
        )

    def make_accountant():
        return dp_accounting.rdp.RdpAccountant(orders)

    accountant = make_accountant()
    accountant.compose(make_event_from_noise(noise_lower_bound))
    initial_epsilon = accountant.get_epsilon(delta)

    # Matches TensorFlow Privacy's reference behavior.
    if initial_epsilon < target_epsilon:
        return 0.0

    target_noise = dp_accounting.calibrate_dp_mechanism(
        make_accountant,
        make_event_from_noise,
        target_epsilon,
        delta,
        dp_accounting.LowerEndpointAndGuess(
            noise_lower_bound,
            noise_lower_bound * 2,
        ),
    )

    return float(target_noise)


def compute_reference_noise_multiplier(dp: DPConfig) -> float:
    """Compute the base noise multiplier used by the reference RING code."""

    dp.validate()

    if not dp.enabled:
        return 0.0

    return _compute_noise_from_budget(
        n=1,
        batch_size=1,
        target_epsilon=dp.epsilon,
        epochs=dp.accountant_epochs,
        delta=dp.delta,
        noise_lower_bound=dp.accountant_tolerance,
    )


def compute_client_noise_stds(
    *,
    client_sample_counts: Sequence[int],
    learning_rate: float,
    dp: DPConfig,
    base_noise_multiplier: Optional[float] = None,
) -> List[float]:
    """Compute per-client noise standard deviations used by RING.

    For client i with n_i samples:

        sigma_i = lr * clip * sqrt(local_epochs) / n_i * noise_multiplier
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
