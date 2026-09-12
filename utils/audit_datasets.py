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
            if not {"split_name", "client_id", "num_examples"}.issubset(meta_cols):
                raise AuditError("client_metadata table is not TFF SqlClientData-compatible")

            example_rows = con.execute(
                "SELECT split_name, COUNT(*), COUNT(DISTINCT client_id) "
                "FROM examples GROUP BY split_name ORDER BY split_name"
            ).fetchall()
            metadata_rows = con.execute(
                "SELECT split_name, COUNT(*), SUM(num_examples) "
                "FROM client_metadata GROUP BY split_name ORDER BY split_name"
            ).fetchall()
            if not example_rows:
                raise AuditError("examples table is empty")

            meta_by_split = {
                str(split): (int(client_count), int(total or 0))
                for split, client_count, total in metadata_rows
            }
            total_samples = 0
            all_client_ids: set[str] = set()
            split_parts: list[str] = []
            for split, count, distinct_clients in example_rows:
                split = str(split)
                count = int(count)
                distinct_clients = int(distinct_clients)
                total_samples += count
                if split not in meta_by_split:
                    raise AuditError(f"metadata missing split {split!r}")
                meta_clients, meta_examples = meta_by_split[split]
                if meta_clients != distinct_clients or meta_examples != count:
                    raise AuditError(
                        f"{split} metadata mismatch: examples={count:,}, "
                        f"clients={distinct_clients:,}; metadata examples={meta_examples:,}, "
                        f"clients={meta_clients:,}"
                    )
                ids = {
                    str(row[0])
                    for row in con.execute(
                        "SELECT DISTINCT client_id FROM examples WHERE split_name = ?",
                        (split,),
                    ).fetchall()
                }
                all_client_ids.update(ids)
                split_parts.append(f"{split} {count:,}/{distinct_clients:,} clients")

            null_proto = int(
                con.execute(
                    "SELECT COUNT(*) FROM examples "
                    "WHERE serialized_example_proto IS NULL OR length(serialized_example_proto)=0"
                ).fetchone()[0]
            )
            if null_proto:
                raise AuditError(f"{null_proto:,} examples have empty serialized payloads")

            counts = [
                int(row[0])
                for row in con.execute(
                    "SELECT SUM(num_examples) FROM client_metadata "
                    "GROUP BY client_id ORDER BY client_id"
                ).fetchall()
            ]
            if not counts or min(counts) <= 0:
                raise AuditError("one or more clients have no examples")

            result.samples = total_samples
            result.clients = len(all_client_ids)
            result.details.extend(
                [
                    "splits: " + " | ".join(split_parts),
                    f"per-client total samples min/median/max {min(counts):,}/{median_int(counts):,}/{max(counts):,}",
                    "FL mapping: client_id selects that client's local examples; split_name selects train/test data",
                ]
            )
    except (AuditError, sqlite3.DatabaseError) as exc:
        result.fail(str(exc))
    return result


def row_is_numeric(row: Sequence[str]) -> bool:
    try:
        for value in row:
            float(value)
        return True
    except ValueError:
        return False


def inspect_csv_layout(path: Path) -> tuple[list[str] | None, int, int]:
    """Return (header_or_none, feature_width, data_rows).

    N-BaIoT CSVs commonly contain a header row followed by numeric feature rows.
    The audit accepts either headered or headerless CSVs, verifies numeric data,
    consistent row width, and returns the number of actual data rows only.
    """
    header: list[str] | None = None
    width: int | None = None
    data_rows = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if width is None:
                    width = len(row)
                    if width <= 0:
                        raise AuditError(f"{path}: no columns")
                    if row_is_numeric(row):
                        data_rows += 1
                    else:
                        header = [str(value).strip() for value in row]
                    continue

                if len(row) != width:
                    raise AuditError(
                        f"{path}: row-width mismatch at line {row_number}: "
                        f"expected {width}, got {len(row)}"
                    )
                if not row_is_numeric(row):
                    raise AuditError(
                        f"{path}: nonnumeric value(s) found in data row {row_number}"
                    )
                data_rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc

    if width is None:
        raise AuditError(f"empty CSV: {path}")
    if data_rows <= 0:
        raise AuditError(f"{path}: contains no numeric data rows")
    return header, width, data_rows


def audit_nbaiot(root: Path, *, fast: bool, test_fraction: float, seed: int) -> AuditResult:
    result = AuditResult("N-BaIoT", partition="natural device")
    dataset_dir = root / "nbaiot"
    result.size_bytes = directory_size(dataset_dir)
    raw = dataset_dir / "raw"
    try:
        if not raw.is_dir():
            raise AuditError(f"missing directory: {raw}")
        leftover_archives = sorted(
            p for p in raw.rglob("*")
            if p.is_file() and p.suffix.casefold() in {".rar", ".zip"}
        )
        if leftover_archives:
            archive_lines = [
                f"{p.relative_to(raw)} ({human_bytes(p.stat().st_size)})"
                for p in leftover_archives
            ]
            preview = "; ".join(archive_lines[:8])
            if len(archive_lines) > 8:
                preview += f"; +{len(archive_lines) - 8} more"
            raise AuditError(
                f"{len(leftover_archives)} compressed archive(s) remain: {preview}. "
                "N-BaIoT is not fully preprocessed; run "
                "`python utils/preprocess_datasets.py nbaiot` and inspect its extraction output."
            )

        device_dirs = sorted(p for p in raw.iterdir() if p.is_dir())
        if not device_dirs:
            raise AuditError("no device directories found")

        if not 0.0 < test_fraction < 1.0:
            raise AuditError("--nbaiot-test-fraction must be between 0 and 1")

        total_rows = 0
        all_widths: set[int] = set()
        canonical_header: tuple[str, ...] | None = None
        headered_files = 0
        headerless_files = 0
        device_summaries: list[str] = []
        for device in device_dirs:
            csv_files = sorted(device.rglob("*.csv"))
            if not csv_files:
                raise AuditError(f"{device.name}: no CSV files")
            benign = [p for p in csv_files if "benign" in p.name.casefold()]
            attacks = [p for p in csv_files if "benign" not in p.name.casefold()]
            if not benign or not attacks:
                raise AuditError(
                    f"{device.name}: requires both benign and attack CSV data"
                )

            device_rows = 0
            benign_rows = 0
            attack_rows = 0
            for csv_path in csv_files:
                header, width, rows = inspect_csv_layout(csv_path)
                all_widths.add(width)
                if header is None:
                    headerless_files += 1
                else:
                    headered_files += 1
                    normalized_header = tuple(header)
                    if canonical_header is None:
                        canonical_header = normalized_header
                    elif normalized_header != canonical_header:
                        raise AuditError(
                            f"{csv_path}: header differs from other N-BaIoT CSV files"
                        )

                if not fast:
                    device_rows += rows
                    if "benign" in csv_path.name.casefold():
                        benign_rows += rows
                    else:
                        attack_rows += rows

            if not fast:
                if benign_rows < 2 or attack_rows < 2:
                    raise AuditError(f"{device.name}: insufficient rows for stratified train/test split")
                benign_test = max(1, int(round(benign_rows * test_fraction)))
                attack_test = max(1, int(round(attack_rows * test_fraction)))
                test_rows = benign_test + attack_test
                train_rows = device_rows - test_rows
                if train_rows <= 0:
                    raise AuditError(f"{device.name}: train split would be empty")
                total_rows += device_rows
                device_summaries.append(
                    f"{device.name} {device_rows:,} "
                    f"(benign {benign_rows:,}/attack {attack_rows:,}; "
                    f"train {train_rows:,}/test {test_rows:,})"
                )
            else:
                device_summaries.append(f"{device.name} {len(csv_files)} files")

        if len(all_widths) != 1:
            raise AuditError(f"inconsistent feature widths: {sorted(all_widths)}")
        if headered_files and headerless_files:
            raise AuditError(
                f"mixed CSV layout: {headered_files} files have headers and "
                f"{headerless_files} do not"
            )

        result.clients = len(device_dirs)
        result.samples = total_rows if not fast else None
        header_state = "headers present and consistent" if headered_files else "headerless numeric CSVs"
        result.details.extend(
            [
                f"features {next(iter(all_widths))} | CSVs {headered_files + headerless_files} | {header_state}",
                "clients: " + "; ".join(device_summaries),
                "labels: CSV filename/path determines benign vs attack class",
                f"FL mapping: one device = one client; deterministic stratified {int((1-test_fraction)*100)}/{int(test_fraction*100)} local train/test split | seed {seed}",
            ]
        )
        if fast:
            result.details.append("sample row counts skipped (--fast mode)")
    except AuditError as exc:
        result.fail(str(exc))
    return result


def selected_datasets(args: argparse.Namespace) -> Sequence[str]:
    if args.datasets == ["all"]:
        return DATASETS
    return tuple(dict.fromkeys(args.datasets))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="+", choices=(*DATASETS, "all"), help="Datasets to audit"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Dataset root (default: repository-level data/)",
    )
    parser.add_argument(
        "--cifar-clients",
        type=int,
        default=10,
        help="Number of simulated CIFAR-10 training clients to validate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic CIFAR-10 client-partition seed to report (default: 42)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip exact N-BaIoT row counts (default performs full count)",
    )
    parser.add_argument(
        "--nbaiot-test-fraction",
        type=float,
        default=0.20,
        help="Per-device stratified test fraction to validate (default: 0.20)",
    )
    args = parser.parse_args()
    if "all" in args.datasets and len(args.datasets) != 1:
        parser.error("Use 'all' alone, or list individual dataset names")
    args.data_dir = args.data_dir.expanduser().resolve()
    return args


def print_report(results: Sequence[AuditResult], root: Path) -> None:
    print("FL DATA READINESS AUDIT")
    print(f"Data root: {root}\n")

    headers = ("Dataset", "Disk", "Samples", "Clients", "Partition", "Status")
    rows = [
        (
            r.dataset,
            human_bytes(r.size_bytes),
            fmt_int(r.samples),
            fmt_int(r.clients),
            r.partition,
            r.status,
        )
        for r in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*(('-' * width) for width in widths)))
    for row in rows:
        print(fmt.format(*row))

    for result in results:
        print(f"\n{result.dataset} [{result.status}]")
        if result.status == "PASS":
            for detail in result.details:
                print(f"  {detail}")
        else:
            for error in result.errors:
                print(f"  ERROR: {error}")

    failed = [r.dataset for r in results if r.status != "PASS"]
    print("\nREADINESS: " + ("PASS - all selected datasets are FL-loadable" if not failed else "FAIL - " + ", ".join(failed)))


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        print(f"ERROR: data directory does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    results: list[AuditResult] = []
    for dataset in selected_datasets(args):
        if dataset == "cifar10":
            results.append(
                audit_cifar10(
                    args.data_dir, cifar_clients=args.cifar_clients, seed=args.seed
                )
            )
        elif dataset == "femnist":
            results.append(
                audit_tff_sqlite(
                    args.data_dir, "femnist", "emnist_all.sqlite", "FEMNIST"
                )
            )
        elif dataset == "shakespeare":
            results.append(
                audit_tff_sqlite(
                    args.data_dir, "shakespeare", "shakespeare.sqlite", "Shakespeare"
                )
            )
        elif dataset == "nbaiot":
            results.append(audit_nbaiot(args.data_dir, fast=args.fast, test_fraction=args.nbaiot_test_fraction, seed=args.seed))

    print_report(results, args.data_dir)
    return 1 if any(r.status != "PASS" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
