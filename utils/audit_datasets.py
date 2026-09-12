#!/usr/bin/env python3
"""Audit installed datasets for federated-learning readiness.

This script is read-only. It does not modify, repartition, normalize, tokenize,
or write dataset files. It verifies the on-disk data produced by
install_datasets.py + preprocess_datasets.py and reports the FL client mapping
that the training loader should use.

Examples:
    python utils/audit_datasets.py all
    python utils/audit_datasets.py cifar10 femnist
    python utils/audit_datasets.py all --cifar-clients 10 --seed 42
    python utils/audit_datasets.py nbaiot

Exit codes:
    0  all selected datasets are ready
    1  one or more datasets failed readiness checks
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - explicit runtime diagnostic
    np = None


DATASETS = ("cifar10", "femnist", "shakespeare", "nbaiot")
MIB = 1024**2
GIB = 1024**3


@dataclass
class AuditResult:
    dataset: str
    size_bytes: int = 0
    samples: int | None = None
    clients: int | None = None
    partition: str = "-"
    status: str = "PASS"
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.status = "FAIL"
        self.errors.append(message)


class AuditError(RuntimeError):
    pass


def human_bytes(size: int) -> str:
    if size >= GIB:
        return f"{size / GIB:.2f} GiB"
    if size >= MIB:
        return f"{size / MIB:.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def fmt_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def median_int(values: Sequence[int]) -> int:
    return int(statistics.median(values)) if values else 0


def ensure_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AuditError(f"missing or empty file: {path}")


def load_cifar_batch(path: Path) -> tuple[int, int]:
    """Return (examples, feature_width) from an official CIFAR Python batch."""
    ensure_file(path)
    if np is None:
        raise AuditError("NumPy is required to inspect CIFAR-10 batches")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
    except Exception as exc:
        raise AuditError(f"cannot read CIFAR batch {path.name}: {exc}") from exc

    data = payload.get(b"data", payload.get("data"))
    labels = payload.get(b"labels", payload.get("labels"))
    if labels is None:
        labels = payload.get(b"fine_labels", payload.get("fine_labels"))
    if data is None or labels is None:
        raise AuditError(f"{path.name} does not contain data/labels")
    if getattr(data, "ndim", None) != 2:
        raise AuditError(f"{path.name} data is not a 2-D array")
    rows, width = int(data.shape[0]), int(data.shape[1])
    if rows != len(labels):
        raise AuditError(
            f"{path.name} has {rows:,} images but {len(labels):,} labels"
        )
    if width != 3072:
        raise AuditError(f"{path.name} feature width is {width}, expected 3072")
    return rows, width


def audit_cifar10(root: Path, *, cifar_clients: int, seed: int) -> AuditResult:
    result = AuditResult("CIFAR-10", partition=f"simulated IID ({cifar_clients})")
    dataset_dir = root / "cifar10"
    result.size_bytes = directory_size(dataset_dir)
    batches = dataset_dir / "raw" / "cifar-10-batches-py"
    try:
        if cifar_clients <= 0:
            raise AuditError("--cifar-clients must be positive")
        train_files = [batches / f"data_batch_{i}" for i in range(1, 6)]
        test_file = batches / "test_batch"
        meta_file = batches / "batches.meta"
        ensure_file(meta_file)
        train_count = sum(load_cifar_batch(path)[0] for path in train_files)
        test_count = load_cifar_batch(test_file)[0]
        if train_count != 50_000 or test_count != 10_000:
            raise AuditError(
                f"unexpected CIFAR-10 counts: train={train_count:,}, test={test_count:,}"
            )

        # Validate the exact deterministic client-index plan without writing it.
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(train_count)
        partitions = np.array_split(shuffled, cifar_clients)
        counts = [int(len(part)) for part in partitions]
        merged = np.concatenate(partitions)
        if len(merged) != train_count or len(np.unique(merged)) != train_count:
            raise AuditError("simulated client assignment has overlap or missing examples")
        if int(merged.min()) != 0 or int(merged.max()) != train_count - 1:
            raise AuditError("simulated client assignment does not cover all training indices")
        if min(counts) <= 0:
            raise AuditError("one or more simulated clients would receive no training data")

        result.samples = train_count + test_count
        result.clients = cifar_clients
        result.details.extend(
            [
                f"train {train_count:,} | test {test_count:,}",
                f"client train samples {min(counts):,}-{max(counts):,} | seed {seed}",
                "FL mapping: deterministic shuffled training indices -> simulated clients; test set stays global",
            ]
        )
    except AuditError as exc:
        result.fail(str(exc))
    return result


def sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    quoted = table.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
    }


def audit_tff_sqlite(root: Path, dataset: str, filename: str, display: str) -> AuditResult:
    result = AuditResult(display, partition="natural client_id")
    dataset_dir = root / dataset
    result.size_bytes = directory_size(dataset_dir)
    path = dataset_dir / "raw" / filename
    try:
        ensure_file(path)
        uri = f"file:{path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            quick = con.execute("PRAGMA quick_check").fetchone()
            if not quick or quick[0] != "ok":
                raise AuditError(f"SQLite integrity check failed: {quick}")

            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            required_tables = {"examples", "client_metadata"}
            if not required_tables.issubset(tables):
                raise AuditError(
                    f"SQLite tables missing: {sorted(required_tables - tables)}"
                )

            ex_cols = sqlite_columns(con, "examples")
            meta_cols = sqlite_columns(con, "client_metadata")
            if not {"split_name", "client_id", "serialized_example_proto"}.issubset(ex_cols):
                raise AuditError("examples table is not TFF SqlClientData-compatible")
