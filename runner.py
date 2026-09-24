#!/usr/bin/env python3
"""End-to-end federated-learning experiment runner.

This runner ties together the current research modules:

    dp.py
    ring_attack.py
    benchmark_backdoor.py
    ProbeGroupDefense.py / ProbeGroupDefense_compact.py

and the repository's dataset/model code.

Supported research datasets:
    cifar10, femnist, shakespeare, nbaiot

Supported scenarios:
    clean       clean FedAvg, no DP
    dp_clean    clean FedAvg with the reference DP noise model
    backdoor    benchmark backdoor + reference DP + FedAvg
    ring        benchmark backdoor + RING coordination + reference DP + FedAvg
    defended    benchmark backdoor + RING + reference DP + ProbeGroup

Examples
--------
Smoke-test the defended CIFAR-10 path:

    python runner_e2e.py --dataset cifar10 --scenario defended --smoke-test

Run the complete scenario suite:

    python runner_e2e.py --dataset cifar10 --suite --rounds 50

Run FEMNIST with two malicious clients:

    python runner_e2e.py --dataset femnist --scenario defended \
        --num-attackers 2 --client-fraction 0.10

Run Shakespeare:

    python runner_e2e.py --dataset shakespeare --scenario defended \
        --batch-size 32 --sequence-trigger-token 1

Important DP note
-----------------
The shared dp.py module reproduces the post-training noise-scale calculation
used by the supplied RING implementation. It explicitly is not a complete
per-example DP-SGD trainer. Accordingly, this runner's DP mode applies the same
reference Gaussian scale to final local model states. This keeps the benign and
RING attacker paths internally consistent for the experiment, but should not be
described as a formal end-to-end DP-SGD implementation.

Data loading
------------
The runner first tries the repository's load_federated_data(...) adapter from:

    utils.load_datasets
    utils.data_setup

Expected return shape:

    train_dataset, test_dataset, client_mapping[, info]

where client_mapping maps client IDs to training indices.

For compatibility with the older runner, it can also load:

    data/<dataset>/cleaned/train.npz
    data/<dataset>/cleaned/test.npz
    data/<dataset>/cleaned/clients.json
    data/<dataset>/cleaned/manifest.json

The repository audit validates the raw source files, while the experiment loader
is responsible for experiment-specific tensorization/tokenization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib
import inspect
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from benchmark_backdoor import PoisonConfig
from dp import (
    DPConfig,
    compute_client_noise_stds,
    compute_reference_noise_multiplier,
)

try:
    from ring_attack import RINGConfig, craft_ring_attack
except ImportError:
    from ring_attack_clean import RINGConfig, craft_ring_attack

try:
    from ProbeGroupDefense import (
        ProbeGroupDefense,
        apply_probe_trigger,
        update_probe_round_summary_metrics,
    )
except ImportError:
    from ProbeGroupDefense_compact import (
        ProbeGroupDefense,
        apply_probe_trigger,
        update_probe_round_summary_metrics,
    )


DATASETS = ("cifar10", "femnist", "shakespeare", "nbaiot")
SCENARIOS = ("clean", "dp_clean", "backdoor", "ring", "defended")
SUITE_SCENARIOS = SCENARIOS


# ---------------------------------------------------------------------------
# Configuration and simple containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    dataset: str
    scenario: str
    rounds: int
    client_fraction: float
    local_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    momentum: float
    seed: int
    device: str
    num_attackers: int
    poison_ratio: float
    target_label: int
    checkpoint_every: int
    dp_enabled: bool
    dp_epsilon: float
    dp_delta: float
    dp_clip: float
    dp_base_noise_multiplier: Optional[float]
    ring_group_size: int
    defense: str
    probe_patch_size: int
    sequence_trigger_token: int
    sequence_trigger_repeat: int
    tabular_trigger_value: float
    tabular_trigger_features: tuple[int, ...]


@dataclass
class ClientResult:
    client_id: Any
    state: Dict[str, torch.Tensor]
    update: Dict[str, torch.Tensor]
    examples: int
    loss: float
    malicious: bool
    poisoned_examples: int = 0


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    examples: int


@dataclass
class DataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    client_ids: list
    client_indices: Dict[Any, list]
    info: Dict[str, Any]
    manifest: Dict[str, Any]
    train_arrays: Optional[Dict[str, torch.Tensor]] = None


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def stable_client_number(client_id: Any, client_order: Mapping[Any, int]) -> int:
    return int(client_order[client_id])


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------

class TensorPairDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def _tensorflow_example_class():
    """Create the minimal TensorFlow Example protobuf type needed by FEMNIST."""

    try:
        from google.protobuf import descriptor_pb2
        from google.protobuf import descriptor_pool
        from google.protobuf import message_factory
    except ImportError as exc:
        raise ImportError(
            "FEMNIST loading requires protobuf: pip install protobuf"
        ) from exc

    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "tensorflow_example.proto"
    file_desc.package = "tensorflow"
    file_desc.syntax = "proto3"

    # BytesList
    message = file_desc.message_type.add()
    message.name = "BytesList"

    field = message.field.add()
    field.name = "value"
    field.number = 1
    field.label = field.LABEL_REPEATED
    field.type = field.TYPE_BYTES

    # FloatList
    message = file_desc.message_type.add()
    message.name = "FloatList"

    field = message.field.add()
    field.name = "value"
    field.number = 1
    field.label = field.LABEL_REPEATED
    field.type = field.TYPE_FLOAT
    field.options.packed = True

    # Int64List
    message = file_desc.message_type.add()
    message.name = "Int64List"

    field = message.field.add()
    field.name = "value"
    field.number = 1
    field.label = field.LABEL_REPEATED
    field.type = field.TYPE_INT64
    field.options.packed = True

    # Feature
    message = file_desc.message_type.add()
    message.name = "Feature"

    oneof = message.oneof_decl.add()
    oneof.name = "kind"

    field = message.field.add()
    field.name = "bytes_list"
    field.number = 1
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.BytesList"
    field.oneof_index = 0

    field = message.field.add()
    field.name = "float_list"
    field.number = 2
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.FloatList"
    field.oneof_index = 0

    field = message.field.add()
    field.name = "int64_list"
    field.number = 3
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.Int64List"
    field.oneof_index = 0

    # Features map
    message = file_desc.message_type.add()
    message.name = "Features"

    entry = message.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True

    field = entry.field.add()
    field.name = "key"
    field.number = 1
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_STRING

    field = entry.field.add()
    field.name = "value"
    field.number = 2
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.Feature"

    field = message.field.add()
    field.name = "feature"
    field.number = 1
    field.label = field.LABEL_REPEATED
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    # Example
    message = file_desc.message_type.add()
    message.name = "Example"

    field = message.field.add()
    field.name = "features"
    field.number = 1
    field.label = field.LABEL_OPTIONAL
    field.type = field.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)

    descriptor = pool.FindMessageTypeByName(
        "tensorflow.Example"
    )

    return message_factory.GetMessageClass(descriptor)


_FEMNIST_EXAMPLE_CLASS = None


def _decode_femnist_example(serialized):
    global _FEMNIST_EXAMPLE_CLASS

    if _FEMNIST_EXAMPLE_CLASS is None:
        _FEMNIST_EXAMPLE_CLASS = _tensorflow_example_class()

    example = _FEMNIST_EXAMPLE_CLASS()
    example.ParseFromString(serialized)

    features = example.features.feature

    pixels = np.asarray(
        features["pixels"].float_list.value,
        dtype=np.float32,
    )

    if pixels.size != 784:
        raise ValueError(
            f"Unexpected FEMNIST pixel count: {pixels.size}"
        )

    labels = features["label"].int64_list.value

    if len(labels) != 1:
        raise ValueError("Invalid FEMNIST label.")

    image = pixels.reshape(1, 28, 28)

    # Official FEMNIST pixels are in [0, 1].
    # Normalize into [-1, 1].
    image = (image - 0.5) / 0.5

    return image, int(labels[0])

def _canonical_client_mapping(mapping) -> tuple[list, Dict[Any, list]]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("The federated data loader returned an empty client mapping.")

    client_ids = list(mapping.keys())
    try:
        client_ids = sorted(client_ids)
    except TypeError:
        client_ids = list(client_ids)

    normalized = {
        client_id: [int(x) for x in list(mapping[client_id])]
        for client_id in client_ids
    }
    return client_ids, normalized


def _load_from_repo_adapter(args) -> Optional[DataBundle]:
    """Use the repository's existing federated-data adapter when available."""

    loader = None
    loader_source = None

    for module_name in ("utils.load_datasets", "utils.data_setup"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        candidate = getattr(module, "load_federated_data", None)
        if candidate is not None:
            loader = candidate
            loader_source = module_name
            break

    if loader is None:
        return None

    loader_args = SimpleNamespace(
        dataset=args.dataset,
        data_dir=str(args.data_dir),
        num_users=args.num_users or {
            "cifar10": 10,
            "femnist": 3400,
            "shakespeare": 715,
            "nbaiot": 3,
        }[args.dataset],
        num_clients=args.num_users or {
            "cifar10": 10,
            "femnist": 3400,
            "shakespeare": 715,
            "nbaiot": 3,
        }[args.dataset],
        cifar_clients=args.num_users or 10,
        iid=args.iid or ("iid" if args.dataset == "cifar10" else "natural"),
        alpha=args.alpha,
        thre_labels=args.thre_labels,
        num_classes=args.num_classes or {
            "cifar10": 10,
            "femnist": 62,
            "shakespeare": 86,
            "nbaiot": 2,
        }[args.dataset],
        seed=args.data_seed,
        test_fraction=args.test_fraction,
        sequence_length=args.sequence_length,
    )

    result = loader(loader_args)

    if not isinstance(result, (tuple, list)) or len(result) not in (3, 4):
        raise TypeError(
            f"{loader_source}.load_federated_data(...) must return "
            "(train, test, clients) or (train, test, clients, info)."
        )

    train_dataset, test_dataset, mapping = result[:3]
    info = dict(result[3]) if len(result) == 4 and isinstance(result[3], Mapping) else {}

    client_ids, client_indices = _canonical_client_mapping(mapping)

    manifest = dict(info.get("manifest", {})) if isinstance(info.get("manifest"), Mapping) else {}
    info.setdefault("loader", loader_source)
    info.setdefault("num_clients", len(client_ids))

    return DataBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_ids=client_ids,
        client_indices=client_indices,
        info=info,
        manifest=manifest,
    )


def _load_from_cleaned_format(args) -> Optional[DataBundle]:
    cleaned = args.data_dir / args.dataset / "cleaned"
    required = {
        name: cleaned / name
        for name in ("train.npz", "test.npz", "clients.json", "manifest.json")
    }

    if not all(path.is_file() for path in required.values()):
        return None

    def load_npz(path: Path):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: torch.from_numpy(archive[name])
                for name in archive.files
            }

        if "x" not in arrays or "y" not in arrays:
            raise ValueError(f"{path} must contain x and y arrays.")

        if args.dataset == "shakespeare":
            arrays["x"] = arrays["x"].long()
            arrays["y"] = arrays["y"].long()
        else:
            arrays["x"] = arrays["x"].float()
            arrays["y"] = arrays["y"].long()

        return arrays

    train = load_npz(required["train.npz"])
    test = load_npz(required["test.npz"])
    clients_json = json.loads(required["clients.json"].read_text(encoding="utf-8"))
    manifest = json.loads(required["manifest.json"].read_text(encoding="utf-8"))

    raw_clients = clients_json.get("clients", clients_json)
    mapping = {}

    for client_id, record in raw_clients.items():
        if isinstance(record, Mapping):
            indices = record.get("train", [])
        else:
            indices = record
        mapping[client_id] = list(indices)

    client_ids, client_indices = _canonical_client_mapping(mapping)

    normalization = manifest.get("normalization")
    if normalization and normalization.get("apply_at_load"):
        mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(normalization["std"], dtype=torch.float32).view(1, -1, 1, 1)
        train["x"] = (train["x"] / 255.0 - mean) / std
        test["x"] = (test["x"] / 255.0 - mean) / std

    return DataBundle(
        train_dataset=TensorPairDataset(train["x"], train["y"]),
        test_dataset=TensorPairDataset(test["x"], test["y"]),
        client_ids=client_ids,
        client_indices=client_indices,
        info={
            "loader": "cleaned_npz",
            "num_clients": len(client_ids),
        },
        manifest=manifest,
        train_arrays=train,
    )

  

def _load_cifar10_audited_raw(args) -> Optional[DataBundle]:
    """Load the repository's audited CIFAR-10 source directly."""

    if args.dataset != "cifar10":
        return None

    raw_root = args.data_dir / "cifar10" / "raw"
    batch_dir = raw_root / "cifar-10-batches-py"

    if not batch_dir.is_dir():
        return None

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Direct CIFAR-10 loading requires torchvision."
        ) from exc

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train = datasets.CIFAR10(
        root=str(raw_root),
        train=True,
        download=False,
        transform=train_transform,
    )
    test = datasets.CIFAR10(
        root=str(raw_root),
        train=False,
        download=False,
        transform=test_transform,
    )

    num_clients = int(args.num_users or 10)
    if num_clients <= 0:
        raise ValueError("--num-users must be positive.")

    rng = np.random.default_rng(int(args.data_seed))
    shuffled = rng.permutation(len(train))
    partitions = np.array_split(shuffled, num_clients)

    client_ids = list(range(num_clients))
    client_indices = {
        client_id: [int(index) for index in partitions[client_id].tolist()]
        for client_id in client_ids
    }

    return DataBundle(
        train_dataset=train,
        test_dataset=test,
        client_ids=client_ids,
        client_indices=client_indices,
        info={
            "loader": "audited_raw_cifar10",
            "name": "cifar10",
            "task": "image_classification",
            "num_classes": 10,
            "num_channels": 3,
            "input_shape": (3, 32, 32),
            "num_clients": num_clients,
            "partition": "deterministic_iid",
            "partition_seed": int(args.data_seed),
        },
        manifest={
            "dataset": "CIFAR-10",
            "num_classes": 10,
            "normalization": {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616],
            },
        },
    )

def _make_femnist_iid_partition(
    labels: torch.Tensor,
    num_clients: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create deterministic stratified IID-style FEMNIST clients."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    labels_np = labels.detach().cpu().numpy()

    rng = np.random.default_rng(seed)

    client_ids = [
        f"iid_client_{index:03d}"
        for index in range(num_clients)
    ]

    partitions = {
        client_id: []
        for client_id in client_ids
    }

    for label in range(62):
        label_indices = np.flatnonzero(
            labels_np == label
        )

        rng.shuffle(label_indices)

        # Rotate the starting client so leftover examples from each
        # class do not always go to the first few clients.
        start_client = int(
            rng.integers(0, num_clients)
        )

        for position, sample_index in enumerate(
            label_indices
        ):
            client_number = (
                start_client + position
            ) % num_clients

            client_id = client_ids[client_number]

            partitions[client_id].append(
                int(sample_index)
            )

    # Shuffle within each simulated client as well.
    for client_id in client_ids:
        client_array = np.asarray(
            partitions[client_id],
            dtype=np.int64,
        )

        rng.shuffle(client_array)

        partitions[client_id] = [
            int(index)
            for index in client_array
        ]

    empty_clients = [
        client_id
        for client_id, indices in partitions.items()
        if not indices
    ]

    if empty_clients:
        raise RuntimeError(
            "IID FEMNIST partition produced empty clients: "
            + ", ".join(empty_clients)
        )

    return partitions

def _load_femnist_audited_raw(args) -> Optional[DataBundle]:
    """Load the audited natural-client FEMNIST SQLite dataset."""

    if args.dataset != "femnist":
        return None

    database = (
        args.data_dir
        / "femnist"
        / "raw"
        / "emnist_all.sqlite"
    )

    if not database.is_file():
        return None

    train_split = "all_train"
    test_split = "all_test"

    num_clients = int(args.num_users or 100)
    
    if num_clients <= 0:
        raise ValueError("--num-users must be positive.")
    
    iid_value = getattr(args, "iid", None)
    
    iid_mode = (
        iid_value is not None
        and str(iid_value).strip().lower()
        in {"1", "true", "yes", "iid"}
    )
    
    if iid_mode:
        source_writer_count = int(
            getattr(
                args,
                "femnist_source_writers",
                100,
            )
        )
    else:
        source_writer_count = num_clients
    
    if source_writer_count <= 0:
        raise ValueError(
            "--femnist-source-writers must be positive."
        )

    max_per_client = int(
        getattr(args, "max_samples_per_client", 0)
    )

    max_test_samples = int(
        getattr(args, "max_test_samples", 0)
    )

    uri = f"file:{database.as_posix()}?mode=ro"

    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    client_indices = {}

    with sqlite3.connect(uri, uri=True) as connection:
        # Select natural writer clients that have examples in both
        # the official training and test splits.
        rows = connection.execute(
            """
            SELECT train.client_id
            FROM client_metadata AS train
            JOIN client_metadata AS test
              ON train.client_id = test.client_id
            WHERE train.split_name = ?
              AND test.split_name = ?
              AND train.num_examples > 0
              AND test.num_examples > 0
            ORDER BY train.client_id
            """,
            (train_split, test_split),
        ).fetchall()

        available_clients = [
            str(row[0])
            for row in rows
        ]

        if not available_clients:
            raise RuntimeError(
                f"No FEMNIST clients found in split {train_split!r}."
            )

        if source_writer_count > len(available_clients):
            raise ValueError(
                f"Requested {source_writer_count} FEMNIST source writers, "
                f"but only {len(available_clients)} are available."
            )

        # Deterministically choose the requested natural writers.
        rng = random.Random(int(args.data_seed))
        rng.shuffle(available_clients)

        selected_clients = available_clients[
            :source_writer_count
        ]

        # ---------------------------------------------------------------
        # Training data
        # ---------------------------------------------------------------

        for client_number, client_id in enumerate(
            selected_clients
        ):
            train_rows = connection.execute(
                """
                SELECT serialized_example_proto
                FROM examples
                WHERE split_name = ?
                  AND client_id = ?
                ORDER BY rowid
                """,
                (train_split, client_id),
            ).fetchall()

            # Optional development-time cap per natural writer.
            if (
                max_per_client > 0
                and len(train_rows) > max_per_client
            ):
                client_rng = np.random.default_rng(
                    int(args.data_seed) + client_number
                )

                chosen = client_rng.choice(
                    len(train_rows),
                    size=max_per_client,
                    replace=False,
                )

                chosen = sorted(
                    int(index)
                    for index in chosen
                )

                train_rows = [
                    train_rows[index]
                    for index in chosen
                ]

            indices = []

            for (serialized,) in train_rows:
                image, label = _decode_femnist_example(
                    serialized
                )

                index = len(train_images)

                train_images.append(image)
                train_labels.append(label)
                indices.append(index)

            if not indices:
                raise RuntimeError(
                    f"FEMNIST client {client_id} "
                    "has no training examples."
                )

            client_indices[client_id] = indices

        # ---------------------------------------------------------------
        # Evaluation data from the same selected writer population
        # ---------------------------------------------------------------

        test_rows = []

        for client_id in selected_clients:
            client_test_rows = connection.execute(
                """
                SELECT serialized_example_proto
                FROM examples
                WHERE split_name = ?
                  AND client_id = ?
                ORDER BY rowid
                """,
                (test_split, client_id),
            ).fetchall()

            test_rows.extend(client_test_rows)

        if not test_rows:
            raise RuntimeError(
                f"No FEMNIST examples found in split {test_split!r}."
            )

        # Cap the test set before decoding it.
        if (
            max_test_samples > 0
            and len(test_rows) > max_test_samples
        ):
            test_rng = np.random.default_rng(
                int(args.data_seed) + 100_000
            )

            chosen = test_rng.choice(
                len(test_rows),
                size=max_test_samples,
                replace=False,
            )

            chosen = sorted(
                int(index)
                for index in chosen
            )

            test_rows = [
                test_rows[index]
                for index in chosen
            ]

        for (serialized,) in test_rows:
            image, label = _decode_femnist_example(
                serialized
            )

            test_images.append(image)
            test_labels.append(label)

    if not train_images:
        raise RuntimeError(
            "FEMNIST loader produced an empty training dataset."
        )

    if not test_images:
        raise RuntimeError(
            "FEMNIST loader produced an empty test dataset."
        )

    # -------------------------------------------------------------------
    # Convert decoded arrays into PyTorch tensors
    # -------------------------------------------------------------------

    train_x = torch.from_numpy(
        np.stack(train_images)
    ).float()

    train_y = torch.tensor(
        train_labels,
        dtype=torch.long,
    )

    test_x = torch.from_numpy(
        np.stack(test_images)
    ).float()

    test_y = torch.tensor(
        test_labels,
        dtype=torch.long,
    )

    if iid_mode:
        client_indices = _make_femnist_iid_partition(
            labels=train_y,
            num_clients=num_clients,
            seed=int(args.data_seed),
        )
    
        client_ids = list(
            client_indices.keys()
        )
    
        partition_name = "iid_stratified_simulated"
    
        print(
            "FEMNIST IID repartition: "
            f"{source_writer_count} source writers -> "
            f"{num_clients} simulated clients"
        )
    else:
        client_ids = list(
            client_indices.keys()
        )
    
        partition_name = "natural_client_id"

    # -------------------------------------------------------------------
    # Label sanity checks
    # -------------------------------------------------------------------

    train_min_label = int(train_y.min().item())
    train_max_label = int(train_y.max().item())

    test_min_label = int(test_y.min().item())
    test_max_label = int(test_y.max().item())

    if train_min_label < 0 or train_max_label >= 62:
        raise ValueError(
            "Unexpected FEMNIST training labels: "
            f"{train_min_label} to {train_max_label}"
        )

    if test_min_label < 0 or test_max_label >= 62:
        raise ValueError(
            "Unexpected FEMNIST test labels: "
            f"{test_min_label} to {test_max_label}"
        )

    # -------------------------------------------------------------------
    # Class-distribution diagnostics
    # -------------------------------------------------------------------

    train_class_counts = torch.bincount(
        train_y,
        minlength=62,
    )

    test_class_counts = torch.bincount(
        test_y,
        minlength=62,
    )

    least_train_classes = torch.argsort(
        train_class_counts
    )[:10]

    most_train_classes = torch.argsort(
        train_class_counts,
        descending=True,
    )[:10]

    print(
        "FEMNIST training class samples: "
        f"min={int(train_class_counts.min())}, "
        f"median={int(train_class_counts.float().median())}, "
        f"max={int(train_class_counts.max())}"
    )

    print(
        "FEMNIST least represented training classes:",
        [
            (
                int(label),
                int(train_class_counts[label]),
            )
            for label in least_train_classes
        ],
    )

    print(
        "FEMNIST most represented training classes:",
        [
            (
                int(label),
                int(train_class_counts[label]),
            )
            for label in most_train_classes
        ],
    )

    print(
        "FEMNIST test class samples: "
        f"min={int(test_class_counts.min())}, "
        f"median={int(test_class_counts.float().median())}, "
        f"max={int(test_class_counts.max())}"
    )

    # -------------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------------

    print(
        f"FEMNIST loaded: "
        f"{len(train_y):,} train, "
        f"{len(test_y):,} test, "
        f"{len(client_ids):,} natural clients"
    )

    print(
        f"FEMNIST labels: "
        f"train={train_min_label}-{train_max_label}, "
        f"test={test_min_label}-{test_max_label}"
    )

    print(
        f"FEMNIST client samples: "
        f"min={min(len(v) for v in client_indices.values())}, "
        f"max={max(len(v) for v in client_indices.values())}"
    )

    return DataBundle(
        train_dataset=TensorPairDataset(
            train_x,
            train_y,
        ),
        test_dataset=TensorPairDataset(
            test_x,
            test_y,
        ),
        client_ids=client_ids,
        client_indices=client_indices,
        info={
            "loader": "audited_raw_femnist",
            "name": "femnist",
            "task": "image_classification",
            "num_classes": 62,
            "num_channels": 1,
            "input_shape": (1, 28, 28),
            "num_clients": len(client_ids),
            "partition": partition_name,
            "iid": iid_mode,
            "source_writers": source_writer_count,
            "partition_seed": int(args.data_seed),
            "train_split": train_split,
            "test_split": test_split,
            "max_samples_per_client": max_per_client,
            "max_test_samples": max_test_samples,
        },
        manifest={
            "dataset": "FEMNIST",
            "num_classes": 62,
            "normalization": {
                "mean": [0.5],
                "std": [0.5],
            },
        },
    )



def load_data(args) -> DataBundle:
    # Direct loaders for the repository's audited raw datasets.
    bundle = _load_cifar10_audited_raw(args)
    if bundle is not None:
        return bundle

    bundle = _load_femnist_audited_raw(args)
    if bundle is not None:
        return bundle

    bundle = _load_from_repo_adapter(args)
    if bundle is not None:
        return bundle

    bundle = _load_from_cleaned_format(args)
    if bundle is not None:
        return bundle

    raise FileNotFoundError(
        "No experiment-ready loader was found for "
        f"{args.dataset!r}."
    )


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

def _model_kwargs(dataset: str, bundle: DataBundle) -> Dict[str, Any]:
    manifest = bundle.manifest
    info = bundle.info

    def infer_num_classes(default):
        for source in (manifest, info):
            if "num_classes" in source:
                return int(source["num_classes"])
            if "vocab_size" in source and dataset == "shakespeare":
                return int(source["vocab_size"])

        # Last resort: inspect targets when the loader exposes them.
        targets = getattr(bundle.train_dataset, "y", None)
        if torch.is_tensor(targets) and targets.numel() > 0:
            return int(targets.max().item()) + 1
        return default

    if dataset == "cifar10":
        return {"num_classes": infer_num_classes(10)}

    if dataset == "femnist":
        return {"num_classes": infer_num_classes(62)}

    if dataset == "shakespeare":
        vocab_size = infer_num_classes(90)
        return {
            "vocab_size": vocab_size,
            "num_classes": vocab_size,
        }

    if dataset == "nbaiot":
        input_dim = info.get("input_dim") or manifest.get("input_dim")
        if input_dim is None:
            sample_x, _ = bundle.train_dataset[0]
            input_dim = int(torch.as_tensor(sample_x).numel())

        return {
            "input_dim": int(input_dim),
            "num_classes": infer_num_classes(2),
        }

    raise ValueError(dataset)


def build_repo_model(dataset: str, bundle: DataBundle):
    """Build the repository model while tolerating the two registry generations.

    Prefer models.py because it contains the revised ResNet/FEMNIST/LSTM/N-BaIoT
    model mapping. Fall back to the older nets.py prepared-data registry.
    """

    kwargs = _model_kwargs(dataset, bundle)
    errors = []

    for module_name in ("models", "nets"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: import failed ({exc})")
            continue

        builder = getattr(module, "build_model", None)
        if builder is None:
            errors.append(f"{module_name}: no build_model")
            continue

        # New registry style: build_model(name, **kwargs)
        try:
            model = builder(dataset, **kwargs)
            if isinstance(model, nn.Module):
                return model
        except (TypeError, KeyError, ValueError) as exc:
            errors.append(f"{module_name} registry call: {exc}")

        # Older prepared-data style: build_model(dataset, train, manifest).
        train_arrays = bundle.train_arrays
        if train_arrays is None:
            # A tiny shape proxy is sufficient for the old builder; it only
            # inspects input dimensions and the maximum target ID.
            sample_x, sample_y = bundle.train_dataset[0]
            x = torch.as_tensor(sample_x)
            y = torch.as_tensor(sample_y)

            if dataset in ("cifar10", "femnist") and x.dim() == 3:
                proxy_x = x.unsqueeze(0)
            elif dataset == "nbaiot":
                proxy_x = x.reshape(1, -1)
            elif dataset == "shakespeare":
                proxy_x = x.unsqueeze(0) if x.dim() == 1 else x
            else:
                proxy_x = x.unsqueeze(0)

            proxy_y = y.reshape(1, *y.shape) if y.dim() > 0 else y.reshape(1)
            train_arrays = {"x": proxy_x, "y": proxy_y}

        try:
            model = builder(dataset, train_arrays, bundle.manifest)
            if isinstance(model, nn.Module):
                return model
        except (TypeError, KeyError, ValueError) as exc:
            errors.append(f"{module_name} prepared-data call: {exc}")

    raise RuntimeError(
        "Unable to construct the repository model for "
        f"{dataset}. Attempts: {'; '.join(errors)}"
    )


# ---------------------------------------------------------------------------
# Model/loss helpers
# ---------------------------------------------------------------------------

def unwrap_logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def task_loss(dataset: str, output, targets: torch.Tensor, pad_token_id: int = -100):
    logits = unwrap_logits(output)

    if dataset == "shakespeare":
        if logits.dim() == 3:
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=pad_token_id,
            )

    return F.cross_entropy(logits, targets.long())


def cpu_state(model: nn.Module):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def state_update(local_state, global_state):
    out = {}
    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            out[key] = local_state[key] - global_state[key]
        else:
            out[key] = torch.zeros_like(global_state[key])
    return out


def weighted_average_updates(updates, sample_counts):
    if not updates:
        raise ValueError("Cannot aggregate zero client updates.")

    weights = np.asarray(sample_counts, dtype=np.float64)
    if weights.sum() <= 0:
        weights[:] = 1.0
    weights /= weights.sum()

    out = {}
    for key in updates[0]:
        first = updates[0][key]
        if not torch.is_floating_point(first):
            out[key] = torch.zeros_like(first)
            continue

        acc = torch.zeros_like(first)
        for weight, update in zip(weights, updates):
            acc += update[key].to(acc.device, acc.dtype) * float(weight)
        out[key] = acc
    return out


def apply_update(model, update):
    state = model.state_dict()
    new_state = {}

    for key, value in state.items():
        if key in update and torch.is_floating_point(value):
            new_state[key] = value + update[key].to(value.device, value.dtype)
        else:
            new_state[key] = value

    model.load_state_dict(new_state)


def round_learning_rate(config: RunConfig, round_idx: int) -> float:
    if config.dataset == "femnist":
        return config.learning_rate

    if config.rounds <= 1:
        return config.learning_rate

    progress = (round_idx - 1) / max(config.rounds - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * cosine


def make_optimizer(model, config: RunConfig, lr: float):
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config.weight_decay,
        )

    raise ValueError(config.optimizer)


# ---------------------------------------------------------------------------
# Benchmark poisoning
# ---------------------------------------------------------------------------

def make_probe_args(
    config: RunConfig,
    bundle: DataBundle,
    output_dir: Path,
    round_idx: int,
    device: torch.device,
):
    num_classes = _model_kwargs(config.dataset, bundle).get(
        "num_classes",
        _model_kwargs(config.dataset, bundle).get("vocab_size", 10),
    )

    return SimpleNamespace(
        dataset=config.dataset,
        device=device,
        seed=config.seed,
        current_round=round_idx,
        save=str(output_dir),
        iid="repository_partition",
        attack_type=config.scenario,
        model="repository_model",
        frac=config.client_fraction,
        num_attacker=config.num_attackers,
        dp_epsilon=config.dp_epsilon if config.dp_enabled else "disabled",
        dp_clip=config.dp_clip if config.dp_enabled else "disabled",
        lr=round_learning_rate(config, round_idx),
        num_classes=int(num_classes),
        probe_target=config.target_label,
        probe_patch_size=config.probe_patch_size,
        probe_trigger_token=config.sequence_trigger_token,
        probe_trigger_repeat=config.sequence_trigger_repeat,
        probe_feature_indices=config.tabular_trigger_features,
        probe_tabular_value=config.tabular_trigger_value,
        probe_hard_drop=False,
        probe_use_median_clip=True,
    )


def poison_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: PoisonConfig,
    dataset: str,
    probe_args,
    seed: int,
):
    """Apply the intentionally obvious benchmark trigger to part of one batch."""

    if config.poison_ratio <= 0 or len(features) == 0:
        return features, targets, 0

    count = min(
        len(features),
        max(1, int(round(config.poison_ratio * len(features)))),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chosen = torch.randperm(len(features), generator=generator)[:count]

    x = features.clone()
    y = targets.clone()

    selected = x[chosen.to(x.device)]
    selected = apply_probe_trigger(selected, probe_args)
    x[chosen.to(x.device)] = selected

    chosen_y = chosen.to(y.device)
    if y.dim() == 1:
        y[chosen_y] = int(config.target_label)
    else:
        # For sequence prediction, poison the final prediction position rather
        # than replacing the entire target sequence.
        y[chosen_y, -1] = int(config.target_label)

    return x, y, count


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_client(
    global_model,
    bundle: DataBundle,
    client_id,
    client_number: int,
    config: RunConfig,
    device,
    round_idx: int,
    malicious: bool,
    poison: PoisonConfig,
    probe_args,
):
    indices = bundle.client_indices[client_id]
    if not indices:
        raise ValueError(f"Client {client_id!r} has no training examples.")

    subset = Subset(bundle.train_dataset, indices)
    batch_size = min(config.batch_size, len(subset))

    generator = torch.Generator()
    generator.manual_seed(config.seed + round_idx * 100_003 + client_number)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    lr = round_learning_rate(config, round_idx)
    optimizer = make_optimizer(local_model, config, lr)

    total_loss = 0.0
    total_examples = 0
    total_poisoned_examples = 0

    pad_token_id = int(
        bundle.manifest.get(
            "pad_token_id",
            bundle.info.get("pad_token_id", -100),
        )
    )

    attack_active = malicious and config.scenario in ("backdoor", "ring", "defended")

    for local_epoch in range(config.local_epochs):
        for batch_idx, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError("Training dataset must yield (features, targets).")

            features, targets = batch[0], batch[1]
            features = features.to(device)
            targets = targets.to(device)

            if attack_active:
                features, targets, poisoned_count = poison_batch(
                    features,
                    targets,
                    config=poison,
                    dataset=config.dataset,
                    probe_args=probe_args,
                    seed=(
                        config.seed
                        + round_idx * 1_000_003
                        + client_number * 1009
                        + local_epoch * 101
                        + batch_idx
                    ),
                )
                total_poisoned_examples += int(poisoned_count)

            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                config.dataset,
                local_model(features),
                targets,
                pad_token_id=pad_token_id,
            )
            loss.backward()
            optimizer.step()

            examples = len(features)
            total_loss += float(loss.detach().cpu()) * examples
            total_examples += examples

    global_state = cpu_state(global_model)
    local_state = cpu_state(local_model)

    result = ClientResult(
        client_id=client_id,
        state=local_state,
        update=state_update(local_state, global_state),
        examples=len(indices),
        loss=total_loss / max(total_examples, 1),
        malicious=malicious,
        poisoned_examples=total_poisoned_examples,
    )

    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Reference DP and RING coordination
# ---------------------------------------------------------------------------

def add_reference_noise_to_state(state, std: float, seed: int):
    if std <= 0:
        return {
            key: value.clone()
            for key, value in state.items()
        }

    out = {}
    for offset, (key, value) in enumerate(state.items()):
        if not torch.is_floating_point(value):
            out[key] = value.clone()
            continue

        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + offset)

        noise = torch.randn(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            generator=generator,
        ) * float(std)

        out[key] = value + noise

    return out


def apply_dp_and_ring(
    client_results: list[ClientResult],
    global_state,
    config: RunConfig,
    round_idx: int,
    dp_config: DPConfig,
    base_noise_multiplier: float,
):
    """Apply independent reference DP noise, or coordinated RING noise."""

    if not dp_config.enabled:
        return client_results, {}

    learning_rate = round_learning_rate(config, round_idx)
    counts = [result.examples for result in client_results]

    stds = compute_client_noise_stds(
        client_sample_counts=counts,
        learning_rate=learning_rate,
        dp=dp_config,
        base_noise_multiplier=base_noise_multiplier,
    )

    ring_mode = config.scenario in ("ring", "defended")
    attacker_positions = [
        i for i, result in enumerate(client_results)
        if result.malicious
    ]

    # Everybody except RING colluders receives independent reference noise.
    ring_position_set = set(attacker_positions) if ring_mode else set()

    for i, result in enumerate(client_results):
        if i in ring_position_set:
            continue

        noised_state = add_reference_noise_to_state(
            result.state,
            std=stds[i],
            seed=config.seed + round_idx * 1_000_033 + i * 997,
        )
        result.state = noised_state
        result.update = state_update(noised_state, global_state)

    metadata = {
        "base_noise_multiplier": float(base_noise_multiplier),
        "noise_stds": [float(x) for x in stds],
        "ring_selected_attackers": len(attacker_positions) if ring_mode else 0,
    }

    if ring_mode and attacker_positions:
        attacker_states = [
            client_results[i].state
            for i in attacker_positions
        ]
        attacker_counts = [
            client_results[i].examples
            for i in attacker_positions
        ]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + round_idx * 2_000_003)

        crafted_states, crafted_updates, ring_metadata = craft_ring_attack(
            attacker_states=attacker_states,
            global_state=global_state,
            client_sample_counts=attacker_counts,
            learning_rate=learning_rate,
            dp=dp_config,
            ring=RINGConfig(
                collusion_group_size=config.ring_group_size,
            ),
            base_noise_multiplier=base_noise_multiplier,
            generator=generator,
        )

        for local_pos, client_pos in enumerate(attacker_positions):
            client_results[client_pos].state = crafted_states[local_pos]
            client_results[client_pos].update = crafted_updates[local_pos]

        metadata["ring"] = ring_metadata

    return client_results, metadata


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_clean(model, bundle: DataBundle, config: RunConfig, device):
    model.eval()
    loader = DataLoader(
        bundle.test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    pad_token_id = int(
        bundle.manifest.get(
            "pad_token_id",
            bundle.info.get("pad_token_id", -100),
        )
    )

    total_loss = 0.0
    loss_denominator = 0
    correct = 0
    denominator = 0
    examples = 0

    for batch in loader:
        features, targets = batch[0].to(device), batch[1].to(device)
        logits = unwrap_logits(model(features))

        loss = task_loss(
            config.dataset,
            logits,
            targets,
            pad_token_id=pad_token_id,
        )

        examples += len(features)

        if config.dataset == "shakespeare" and logits.dim() == 3:
            predictions = logits.argmax(dim=-1)
            mask = (
                targets.ne(pad_token_id)
                if pad_token_id >= 0
                else torch.ones_like(targets, dtype=torch.bool)
            )
            valid = int(mask.sum().item())
            correct += int(predictions.eq(targets).logical_and(mask).sum().item())
            denominator += valid
            total_loss += float(loss) * valid
            loss_denominator += valid
        else:
            predictions = logits.argmax(dim=-1)
            correct += int(predictions.eq(targets).sum().item())
            denominator += int(targets.numel())
            total_loss += float(loss) * len(features)
            loss_denominator += len(features)

    return EvalResult(
        loss=total_loss / max(loss_denominator, 1),
        accuracy=correct / max(denominator, 1),
        examples=examples,
    )


@torch.inference_mode()
def evaluate_backdoor_diagnostics(model, bundle, config, device, probe_args):
    """Measure whether the trigger itself causes targeted misclassification.

    Metrics:
      asr:
        Standard targeted attack success rate on triggered, non-target examples.
      clean_target_rate:
        How often the clean model already predicts the attack target on the same
        eligible examples. This is the baseline that raw ASR does not reveal.
      conditional_asr:
        Among eligible examples that the clean model classified correctly, how
        often adding the trigger flips the prediction to the attack target.
      target_confidence_lift:
        Mean increase in attack-target softmax probability caused by the trigger.
    """

    if config.scenario in ("clean", "dp_clean"):
        return None

    model.eval()
    loader = DataLoader(
        bundle.test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    target_label = int(config.target_label)

    eligible = 0
    clean_target_count = 0
    triggered_target_count = 0
    clean_correct_eligible = 0
    conditional_successes = 0
    clean_target_confidence_sum = 0.0
    triggered_target_confidence_sum = 0.0

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError("Evaluation dataset must yield (features, targets).")

        features = batch[0].to(device)
        targets = batch[1].to(device)
        triggered = apply_probe_trigger(features, probe_args)

        clean_logits = unwrap_logits(model(features))
        triggered_logits = unwrap_logits(model(triggered))

        if config.dataset == "shakespeare" and clean_logits.dim() == 3:
            clean_logits = clean_logits[:, -1, :]
            triggered_logits = triggered_logits[:, -1, :]
            original_target = targets[:, -1] if targets.dim() > 1 else targets
        else:
            original_target = targets

        clean_predictions = clean_logits.argmax(dim=-1)
        triggered_predictions = triggered_logits.argmax(dim=-1)

        if original_target.ndim > 1:
            original_target = original_target.reshape(-1)
        if clean_predictions.ndim > 1:
            clean_predictions = clean_predictions.reshape(-1)
        if triggered_predictions.ndim > 1:
            triggered_predictions = triggered_predictions.reshape(-1)

        mask = original_target.ne(target_label)
        batch_eligible = int(mask.sum().item())
        if batch_eligible == 0:
            continue

        clean_probabilities = torch.softmax(clean_logits, dim=-1)
        triggered_probabilities = torch.softmax(triggered_logits, dim=-1)

        eligible += batch_eligible
        clean_target_count += int(
            (clean_predictions.eq(target_label) & mask).sum().item()
        )
        triggered_target_count += int(
            (triggered_predictions.eq(target_label) & mask).sum().item()
        )

        clean_correct_mask = clean_predictions.eq(original_target) & mask
        clean_correct_eligible += int(clean_correct_mask.sum().item())
        conditional_successes += int(
            (triggered_predictions.eq(target_label) & clean_correct_mask).sum().item()
        )

        clean_target_confidence_sum += float(
            clean_probabilities[mask, target_label].sum().item()
        )
        triggered_target_confidence_sum += float(
            triggered_probabilities[mask, target_label].sum().item()
        )

    if eligible == 0:
        return {
            "asr": 0.0,
            "clean_target_rate": 0.0,
            "conditional_asr": 0.0,
            "clean_target_confidence": 0.0,
            "triggered_target_confidence": 0.0,
            "target_confidence_lift": 0.0,
            "eligible_examples": 0,
            "conditional_examples": 0,
        }

    clean_target_rate = clean_target_count / eligible
    asr = triggered_target_count / eligible
    conditional_asr = conditional_successes / max(clean_correct_eligible, 1)
    clean_target_confidence = clean_target_confidence_sum / eligible
    triggered_target_confidence = triggered_target_confidence_sum / eligible

    return {
        "asr": asr,
        "clean_target_rate": clean_target_rate,
        "conditional_asr": conditional_asr,
        "clean_target_confidence": clean_target_confidence,
        "triggered_target_confidence": triggered_target_confidence,
        "target_confidence_lift": (
            triggered_target_confidence - clean_target_confidence
        ),
        "eligible_examples": eligible,
        "conditional_examples": clean_correct_eligible,
    }


# ---------------------------------------------------------------------------
# Experiment engine
# ---------------------------------------------------------------------------

class Experiment:
    def __init__(
        self,
        *,
        config: RunConfig,
        bundle: DataBundle,
        model: nn.Module,
        output_dir: Path,
        args,
    ):
        self.config = config
        self.bundle = bundle
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.output_dir = output_dir
        self.args = args

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client_order = {
            client_id: i
            for i, client_id in enumerate(bundle.client_ids)
        }

        if config.num_attackers > len(bundle.client_ids):
            raise ValueError(
                f"num_attackers={config.num_attackers} exceeds "
                f"{len(bundle.client_ids)} total clients."
            )

        attacker_rng = random.Random(config.seed + 424_242)
        self.malicious_clients = set(
            attacker_rng.sample(
                bundle.client_ids,
                config.num_attackers,
            )
        ) if config.num_attackers > 0 else set()

        self.poison = PoisonConfig(
            poison_ratio=config.poison_ratio,
            target_label=config.target_label,
            seed=config.seed,
        )

        self.dp = DPConfig(
            enabled=config.dp_enabled,
            epsilon=config.dp_epsilon,
            delta=config.dp_delta,
            clip=config.dp_clip,
            local_epochs=config.local_epochs,
            total_fl_epochs=config.rounds,
            client_fraction=config.client_fraction,
        )

        if not self.dp.enabled:
            self.base_noise_multiplier = 0.0
        elif config.dp_base_noise_multiplier is not None:
            self.base_noise_multiplier = float(config.dp_base_noise_multiplier)
        else:
            try:
                self.base_noise_multiplier = compute_reference_noise_multiplier(self.dp)
            except ImportError as exc:
                raise RuntimeError(
                    "DP/RING scenarios need dp-accounting to compute the "
                    "reference noise multiplier. Install it with "
                    "'pip install dp-accounting' or pass a precomputed "
                    "--dp-base-noise-multiplier."
                ) from exc

        (self.output_dir / "config.json").write_text(
            json.dumps(
                {
                    "run": asdict(config),
                    "data_info": bundle.info,
                    "manifest": bundle.manifest,
                    "dp_reference_noise_multiplier": self.base_noise_multiplier,
                    "dp_mode": "reference_posttraining" if self.dp.enabled else "disabled",
                },
                indent=2,
                sort_keys=True,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

        (self.output_dir / "malicious_clients.json").write_text(
            json.dumps(
                [str(x) for x in sorted(self.malicious_clients, key=str)],
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    def selected_clients(self, round_idx: int):
        count = max(
            1,
            int(round(len(self.bundle.client_ids) * self.config.client_fraction)),
        )
        count = min(count, len(self.bundle.client_ids))

        rng = random.Random(self.config.seed + round_idx)
        return sorted(
            rng.sample(self.bundle.client_ids, count),
            key=lambda x: self.client_order[x],
        )

    def write_metric(self, row):
        path = self.output_dir / "metrics.csv"
        write_header = not path.exists()

        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def checkpoint(self, round_idx):
        path = self.output_dir / f"checkpoint_round_{round_idx:04d}.pt"
        temp = path.with_suffix(".tmp")
        torch.save(
            {
                "round": int(round_idx),
                "model": cpu_state(self.model),
                "scenario": self.config.scenario,
            },
            temp,
        )
        os.replace(temp, path)

    def run(self):
        clean_initial = evaluate_clean(
            self.model,
            self.bundle,
            self.config,
            self.device,
        )

        print(
            f"[{self.config.scenario}] initial "
            f"loss={clean_initial.loss:.4f} "
            f"accuracy={clean_initial.accuracy:.4f}"
        )

        global_attacker_numeric_ids = [
            self.client_order[client_id]
            for client_id in self.malicious_clients
        ]

        for round_idx in range(1, self.config.rounds + 1):
            started = time.monotonic()
            selected = self.selected_clients(round_idx)

            probe_args = make_probe_args(
                self.config,
                self.bundle,
                self.output_dir,
                round_idx,
                self.device,
            )

            results = []

            for client_id in selected:
                result = train_client(
                    self.model,
                    self.bundle,
                    client_id,
                    stable_client_number(client_id, self.client_order),
                    self.config,
                    self.device,
                    round_idx,
                    client_id in self.malicious_clients,
                    self.poison,
                    probe_args,
                )
                results.append(result)

            global_state = cpu_state(self.model)

            results, dp_metadata = apply_dp_and_ring(
                results,
                global_state,
                self.config,
                round_idx,
                self.dp,
                self.base_noise_multiplier,
            )

            updates = [result.update for result in results]
            sample_counts = [result.examples for result in results]

            if self.config.defense == "probegroup":
                selected_numeric_ids = [
                    self.client_order[result.client_id]
                    for result in results
                ]

                defended_state = ProbeGroupDefense(
                    w_list=[result.state for result in results],
                    w_updates=updates,
                    global_model=self.model,
                    dataset_test=self.bundle.test_dataset,
                    args=probe_args,
                    per_run=self.config.seed,
                    first_call=round_idx == 1,
                    w_length=sample_counts,
                    users_idx=selected_numeric_ids,
                    idx_attacker=global_attacker_numeric_ids,
                    debug=False,
                )

                aggregated_update = state_update(
                    {
                        key: value.detach().cpu()
                        for key, value in defended_state.items()
                    },
                    global_state,
                )
            else:
                aggregated_update = weighted_average_updates(
                    updates,
                    sample_counts,
                )

            apply_update(self.model, aggregated_update)

            evaluation = evaluate_clean(
                self.model,
                self.bundle,
                self.config,
                self.device,
            )

            attack_metrics = evaluate_backdoor_diagnostics(
                self.model,
                self.bundle,
                self.config,
                self.device,
                probe_args,
            )
            asr = None if attack_metrics is None else attack_metrics["asr"]

            avg_train_loss = (
                sum(r.loss * r.examples for r in results)
                / max(sum(r.examples for r in results), 1)
            )

            selected_attackers = sum(r.malicious for r in results)
            elapsed = time.monotonic() - started

            row = {
                "round": round_idx,
                "scenario": self.config.scenario,
                "defense": self.config.defense,
                "dp_enabled": int(self.config.dp_enabled),
                "learning_rate": f"{round_learning_rate(self.config, round_idx):.10g}",
                "train_loss": f"{avg_train_loss:.8f}",
                "test_loss": f"{evaluation.loss:.8f}",
                "clean_accuracy": f"{evaluation.accuracy:.8f}",
                "asr": "" if asr is None else f"{asr:.8f}",
                "conditional_asr": (
                    ""
                    if attack_metrics is None
                    else f"{attack_metrics['conditional_asr']:.8f}"
                ),
                "clean_target_rate": (
                    ""
                    if attack_metrics is None
                    else f"{attack_metrics['clean_target_rate']:.8f}"
                ),
                "clean_target_confidence": (
                    ""
                    if attack_metrics is None
                    else f"{attack_metrics['clean_target_confidence']:.8f}"
                ),
                "triggered_target_confidence": (
                    ""
                    if attack_metrics is None
                    else f"{attack_metrics['triggered_target_confidence']:.8f}"
                ),
                "target_confidence_lift": (
                    ""
                    if attack_metrics is None
                    else f"{attack_metrics['target_confidence_lift']:.8f}"
                ),
                "attack_eligible_examples": (
                    ""
                    if attack_metrics is None
                    else int(attack_metrics["eligible_examples"])
                ),
                "conditional_examples": (
                    ""
                    if attack_metrics is None
                    else int(attack_metrics["conditional_examples"])
                ),
                "poisoned_examples": sum(
                    int(result.poisoned_examples) for result in results
                ),
                "selected_clients": len(selected),
                "selected_attackers": selected_attackers,
                "total_attackers": len(self.malicious_clients),
                "ring_selected_attackers": int(
                    dp_metadata.get("ring_selected_attackers", 0)
                ),
                "seconds": f"{elapsed:.3f}",
            }
            self.write_metric(row)

            if self.config.defense == "probegroup":
                update_probe_round_summary_metrics(
                    args=probe_args,
                    per_run=self.config.seed,
                    round_idx=round_idx,
                    asr=asr,
                    clean_acc=evaluation.accuracy,
                    test_loss=evaluation.loss,
                    avg_train_loss=avg_train_loss,
                )

            asr_text = "-" if asr is None else f"{asr:.4f}"
            conditional_asr_text = (
                "-"
                if attack_metrics is None
                else f"{attack_metrics['conditional_asr']:.4f}"
            )
            confidence_lift_text = (
                "-"
                if attack_metrics is None
                else f"{attack_metrics['target_confidence_lift']:.4f}"
            )
            poisoned_examples = sum(
                int(result.poisoned_examples) for result in results
            )

            print(
                f"[{self.config.scenario}] round {round_idx:03d}: "
                f"train_loss={avg_train_loss:.4f} "
                f"clean_acc={evaluation.accuracy:.4f} "
                f"asr={asr_text} "
                f"conditional_asr={conditional_asr_text} "
                f"confidence_lift={confidence_lift_text} "
                f"poisoned={poisoned_examples} "
                f"clients={len(selected)} attackers={selected_attackers} "
                f"time={elapsed:.1f}s"
            )

            if (
                self.config.checkpoint_every > 0
                and round_idx % self.config.checkpoint_every == 0
            ):
                self.checkpoint(round_idx)

        self.checkpoint(self.config.rounds)


# ---------------------------------------------------------------------------
# CLI / scenario resolution
# ---------------------------------------------------------------------------

def parse_args():
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--scenario", choices=SCENARIOS, default="defended")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run clean, dp_clean, backdoor, ring, and defended scenarios sequentially.",
    )

    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results")
    parser.add_argument("--run-name")

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-fraction", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--target-label", type=int, default=1)

    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip", type=float, default=1.0)
    parser.add_argument(
        "--dp-base-noise-multiplier",
        type=float,
        help=(
            "Optional precomputed reference DP noise multiplier. "
            "When omitted, dp.py computes it with dp-accounting."
        ),
    )
    parser.add_argument("--ring-group-size", type=int, default=-1)

    parser.add_argument("--probe-patch-size", type=int, default=6)
    parser.add_argument("--sequence-trigger-token", type=int, default=1)
    parser.add_argument("--sequence-trigger-repeat", type=int, default=3)
    parser.add_argument("--tabular-trigger-value", type=float, default=999.0)
    parser.add_argument(
        "--tabular-trigger-features",
        default="0,1,2",
        help="Comma-separated N-BaIoT trigger feature indices.",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=5)

    # Compatibility arguments for repository data adapters.
    parser.add_argument("--num-users", type=int)
    parser.add_argument("--iid", default=None)
    parser.add_argument(
        "--femnist-source-writers",
        type=int,
        default=100,
        help=(
            "Number of natural FEMNIST writers used to construct "
            "the pooled dataset before IID repartitioning."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--thre-labels", type=int, default=2)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument(
        "--max-samples-per-client",
        type=int,
        default=0,
        help=(
            "Maximum training examples retained per client. "
            "0 uses all available examples."
        ),
    )
    
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help=(
            "Maximum global test examples retained. "
            "0 uses all available examples."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 rounds and force at least two selected clients.",
    )

    return parser.parse_args()


def scenario_config(args, bundle: DataBundle, scenario: str) -> RunConfig:
    if scenario == "clean":
        attack = False
        dp_enabled = False
        defense = "fedavg"
    elif scenario == "dp_clean":
        attack = False
        dp_enabled = True
        defense = "fedavg"
    elif scenario == "backdoor":
        attack = True
        dp_enabled = True
        defense = "fedavg"
    elif scenario == "ring":
        attack = True
        dp_enabled = True
        defense = "fedavg"
    elif scenario == "defended":
        attack = True
        dp_enabled = True
        defense = "probegroup"
    else:
        raise ValueError(scenario)

    rounds = 2 if args.smoke_test else args.rounds
    fraction = args.client_fraction

    if args.smoke_test:
        fraction = max(
            fraction,
            min(1.0, 2.0 / max(len(bundle.client_ids), 1)),
        )

    optimizer = args.optimizer or (
        "sgd" if args.dataset in ("cifar10", "femnist") else "adam"
    )

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = {
            "cifar10": 0.1,
            "femnist": 0.05,
            "shakespeare": 2e-3,
            "nbaiot": 1e-3,
        }[args.dataset]

    num_attackers = args.num_attackers if attack else 0

    if scenario in ("ring", "defended") and num_attackers < 2:
        raise ValueError("RING scenarios require at least two configured malicious clients.")

    return RunConfig(
        dataset=args.dataset,
        scenario=scenario,
        rounds=rounds,
        client_fraction=fraction,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=float(learning_rate),
        optimizer=optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        seed=args.seed,
        device=args.device,
        num_attackers=num_attackers,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        checkpoint_every=args.checkpoint_every,
        dp_enabled=dp_enabled,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_clip=args.dp_clip,
        dp_base_noise_multiplier=args.dp_base_noise_multiplier,
        ring_group_size=args.ring_group_size,
        defense=defense,
        probe_patch_size=args.probe_patch_size,
        sequence_trigger_token=args.sequence_trigger_token,
        sequence_trigger_repeat=args.sequence_trigger_repeat,
        tabular_trigger_value=args.tabular_trigger_value,
        tabular_trigger_features=tuple(
            int(value.strip())
            for value in args.tabular_trigger_features.split(",")
            if value.strip()
        ),
    )


def validate_args(args):
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive.")
    if args.local_epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--local-epochs and --batch-size must be positive.")
    if not 0 < args.client_fraction <= 1:
        raise SystemExit("--client-fraction must be in (0, 1].")
    if not 0 <= args.poison_ratio <= 1:
        raise SystemExit("--poison-ratio must be in [0, 1].")
    if args.num_attackers < 0:
        raise SystemExit("--num-attackers cannot be negative.")
    if args.dp_epsilon <= 0 or args.dp_clip <= 0:
        raise SystemExit("--dp-epsilon and --dp-clip must be positive.")
    if args.probe_patch_size <= 0 or args.sequence_trigger_repeat <= 0:
        raise SystemExit("--probe-patch-size and --sequence-trigger-repeat must be positive.")
    if not 0 < args.dp_delta < 1:
        raise SystemExit("--dp-delta must be in (0, 1).")
    if (
        args.dp_base_noise_multiplier is not None
        and args.dp_base_noise_multiplier < 0
    ):
        raise SystemExit("--dp-base-noise-multiplier cannot be negative.")


def main():
    args = parse_args()
    validate_args(args)

    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    set_seed(args.seed)
    bundle = load_data(args)

    print(
        f"dataset={args.dataset} "
        f"clients={len(bundle.client_ids)} "
        f"loader={bundle.info.get('loader', 'repository')}"
    )

    scenarios = SUITE_SCENARIOS if args.suite else (args.scenario,)

    for scenario in scenarios:
        config = scenario_config(args, bundle, scenario)

        set_seed(config.seed)
        model = build_repo_model(config.dataset, bundle)

        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        if args.run_name:
            run_name = (
                f"{args.run_name}_{scenario}"
                if args.suite
                else args.run_name
            )
        else:
            run_name = f"{scenario}_seed{config.seed}"
        destination = (
            args.output_dir
            / config.dataset
            / run_name
        )

        print(
            f"\n=== {scenario} ===\n"
            f"model_parameters={parameters:,} "
            f"rounds={config.rounds} "
            f"client_fraction={config.client_fraction:.3f} "
            f"attackers={config.num_attackers} "
            f"defense={config.defense} "
            f"dp={config.dp_enabled}\n"
            f"results={destination}"
        )

        experiment = Experiment(
            config=config,
            bundle=bundle,
            model=model,
            output_dir=destination,
            args=args,
        )
        experiment.run()


if __name__ == "__main__":
    main()
