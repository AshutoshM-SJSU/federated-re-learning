"""
benchmark_backdoor.py

Simple, intentionally conspicuous poisoning utilities for controlled
federated-learning security experiments.

Supported modalities:
- CIFAR-10 / FEMNIST: visible image patch + target-label assignment
- Shakespeare: obvious token/character trigger + target-label assignment
- N-BaIoT: fixed synthetic feature pattern + target-label assignment

These utilities are designed for benchmark construction and detection studies,
not for stealth or evasion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class PoisonConfig:
    poison_ratio: float = 0.1
    target_label: int = 0
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 <= self.poison_ratio <= 1.0:
            raise ValueError("poison_ratio must be in [0, 1]")


def sample_poison_indices(
    n: int,
    poison_ratio: float,
    seed: int = 42,
) -> set[int]:
    if n <= 0:
        return set()
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be in [0, 1]")

    count = min(int(poison_ratio * n), n)
    if count == 0:
        return set()

    rng = np.random.RandomState(seed)
    selected = rng.choice(n, count, replace=False)
    return set(int(i) for i in selected.tolist())


# ---------------------------------------------------------------------------
# IMAGE BENCHMARKS: CIFAR-10 and FEMNIST
# ---------------------------------------------------------------------------

def apply_visible_image_patch(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 6,
    value: int = 255,
    location: str = "top_right",
) -> np.ndarray:
    """
    Add a large, obvious square patch to an image.

    Works with:
    - H x W grayscale arrays
    - H x W x C RGB arrays
    """
    arr = np.array(image, copy=True)

    if arr.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")

    h, w = arr.shape[:2]
    size = min(patch_size, h, w)

    if location == "top_right":
        r0, r1 = 0, size
        c0, c1 = w - size, w
    elif location == "top_left":
        r0, r1 = 0, size
        c0, c1 = 0, size
    elif location == "bottom_right":
        r0, r1 = h - size, h
        c0, c1 = w - size, w
    elif location == "bottom_left":
        r0, r1 = h - size, h
        c0, c1 = 0, size
    else:
        raise ValueError("unsupported location")

    if arr.ndim == 2:
        arr[r0:r1, c0:c1] = value
    else:
        arr[r0:r1, c0:c1, :] = value

    return arr


class ImagePoisonDataset(Dataset):
    """
    Generic wrapper for image datasets such as CIFAR-10 and FEMNIST.

    Poisoned samples receive:
      1. a conspicuous fixed image patch
      2. the configured target label
    """

    def __init__(
        self,
        dataset: Dataset,
        config: PoisonConfig,
        patch_size: int = 6,
        patch_value: int = 255,
        patch_location: str = "top_right",
        transform=None,
    ):
        config.validate()
        self.dataset = dataset
        self.config = config
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.patch_location = patch_location
        self.transform = transform

        self.poison_indices = sample_poison_indices(
            len(dataset),
            config.poison_ratio,
            config.seed,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if index in self.poison_indices:
            image = apply_visible_image_patch(
                image,
                patch_size=self.patch_size,
                value=self.patch_value,
                location=self.patch_location,
            )
            label = self.config.target_label

        if self.transform is not None:
            if isinstance(image, np.ndarray):
                if image.ndim == 3:
                    image = Image.fromarray(image.astype(np.uint8), mode="RGB")
                else:
                    image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# SHAKESPEARE BENCHMARK
# ---------------------------------------------------------------------------

def apply_visible_sequence_trigger(
    sequence: Sequence[int],
    trigger_token: int,
    repeat: int = 3,
) -> List[int]:
    """
    Insert an obvious repeated trigger token at the beginning of a sequence.

    The trigger is intentionally simple and conspicuous.
    """
    seq = list(sequence)
    return [int(trigger_token)] * int(repeat) + seq


class ShakespearePoisonDataset(Dataset):
    """
    Wrapper for character/token sequence datasets.

    Expected base item format:
        (input_sequence, target)

    input_sequence may be a tensor, list, tuple, or NumPy array.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: PoisonConfig,
        trigger_token: int,
        repeat: int = 3,
    ):
        config.validate()
        self.dataset = dataset
        self.config = config
        self.trigger_token = int(trigger_token)
        self.repeat = int(repeat)

        self.poison_indices = sample_poison_indices(
            len(dataset),
            config.poison_ratio,
            config.seed,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sequence, target = self.dataset[index]

        is_tensor = torch.is_tensor(sequence)
        dtype = sequence.dtype if is_tensor else None
        device = sequence.device if is_tensor else None

        seq_list = sequence.tolist() if is_tensor else list(sequence)

        if index in self.poison_indices:
            seq_list = apply_visible_sequence_trigger(
                seq_list,
                trigger_token=self.trigger_token,
                repeat=self.repeat,
            )

            # Preserve the original sequence length for fixed-length models.
            original_len = len(sequence)
            seq_list = seq_list[:original_len]
            target = self.config.target_label

        if is_tensor:
            sequence = torch.tensor(seq_list, dtype=dtype, device=device)
        else:
            sequence = np.asarray(seq_list, dtype=np.int64)

        return sequence, target


# ---------------------------------------------------------------------------
# N-BAIOT BENCHMARK
# ---------------------------------------------------------------------------

def apply_visible_tabular_trigger(
    features: Union[np.ndarray, torch.Tensor, Sequence[float]],
    feature_indices: Sequence[int] = (0, 1, 2),
    trigger_value: float = 999.0,
):
    """
    Set a small, fixed feature subset to an extreme synthetic value.

    This is deliberately obvious so it can serve as a simple detection target.
    """
    is_tensor = torch.is_tensor(features)

    if is_tensor:
        out = features.clone()
        for idx in feature_indices:
            out[int(idx)] = float(trigger_value)
        return out

    out = np.array(features, dtype=np.float32, copy=True)
    for idx in feature_indices:
        out[int(idx)] = float(trigger_value)
    return out


class NBaIoTPoisonDataset(Dataset):
    """
    Wrapper for tabular N-BaIoT-style datasets.

    Expected base item format:
        (feature_vector, label)
    """

    def __init__(
        self,
        dataset: Dataset,
        config: PoisonConfig,
        feature_indices: Sequence[int] = (0, 1, 2),
        trigger_value: float = 999.0,
    ):
        config.validate()
        self.dataset = dataset
        self.config = config
        self.feature_indices = tuple(int(i) for i in feature_indices)
        self.trigger_value = float(trigger_value)

        self.poison_indices = sample_poison_indices(
            len(dataset),
            config.poison_ratio,
            config.seed,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, label = self.dataset[index]

        if index in self.poison_indices:
            features = apply_visible_tabular_trigger(
                features,
                feature_indices=self.feature_indices,
                trigger_value=self.trigger_value,
            )
            label = self.config.target_label

        return features, label


# ---------------------------------------------------------------------------
# FEDERATED-CLIENT HELPER
# ---------------------------------------------------------------------------

def poison_client_indices(
    client_indices: Iterable[int],
    poison_ratio: float,
    seed: int = 42,
) -> set[int]:
    """
    Select poisoned examples from one client's existing index set.

    Useful when the federated pipeline stores client partitions separately
    instead of wrapping one Dataset per client.
    """
    idxs = list(int(i) for i in client_indices)

    if not idxs:
        return set()

    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be in [0, 1]")

    count = min(int(poison_ratio * len(idxs)), len(idxs))
    if count == 0:
        return set()

    rng = np.random.RandomState(seed)
    selected = rng.choice(idxs, count, replace=False)
    return set(int(i) for i in selected.tolist())


__all__ = [
    "PoisonConfig",
    "sample_poison_indices",
    "apply_visible_image_patch",
    "ImagePoisonDataset",
    "apply_visible_sequence_trigger",
    "ShakespearePoisonDataset",
    "apply_visible_tabular_trigger",
    "NBaIoTPoisonDataset",
    "poison_client_indices",
]
