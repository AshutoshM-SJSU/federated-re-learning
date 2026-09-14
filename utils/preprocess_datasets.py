#!/usr/bin/env python3
"""Prepare installed federated-learning datasets in place.

This script intentionally performs only storage/data-hygiene preprocessing:
- decompress/extract downloaded archives,
- validate expected dataset structure,
- validate SQLite and CSV readability,
- remove source archives after successful extraction/decompression by default.

It does NOT normalize features, tokenize text, create batches, create simulated
clients, repartition naturally federated datasets, or construct model tensors.
Those experiment-specific operations should happen when the FL dataset is loaded.

Examples:
    python utils/preprocess_datasets.py all
    python utils/preprocess_datasets.py cifar10 femnist
    python utils/preprocess_datasets.py shakespeare
    python utils/preprocess_datasets.py nbaiot
    python utils/preprocess_datasets.py all --keep-archives
    python utils/preprocess_datasets.py all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import lzma
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


DATASETS = ("cifar10", "femnist", "shakespeare", "nbaiot")
COPY_CHUNK = 1024 * 1024


class PreprocessError(RuntimeError):
    """Raised when a dataset cannot be safely prepared."""


def phase(message: str) -> None:
    print(f"\n==> {message}")


def safe_relative_path(name: str) -> Path:
    """Return a safe relative archive path or raise."""
    normalized = name.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise PreprocessError(f"Unsafe archive member path: {name!r}")
    parts = [part for part in parsed.parts if part not in ("", ".")]
    if not parts:
        raise PreprocessError(f"Empty archive member path: {name!r}")
    return Path(*parts)


def atomic_replace_from_stream(reader, destination: Path) -> None:
    """Write reader contents to destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            while True:
                chunk = reader.read(COPY_CHUNK)
                if not chunk:
                    break
                output.write(chunk)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def remove_file(path: Path, keep_archives: bool) -> None:
    if keep_archives:
        return
    path.unlink(missing_ok=True)


def validate_sqlite(path: Path, required_table: str | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreprocessError(f"SQLite file is missing or empty: {path}")
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise PreprocessError(f"SQLite integrity check failed for {path}: {result}")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if not tables:
                raise PreprocessError(f"SQLite database contains no tables: {path}")
            if required_table and required_table not in tables:
                raise PreprocessError(
                    f"SQLite database {path} does not contain expected table "
                    f"{required_table!r}; found {sorted(tables)}"
                )
    except sqlite3.DatabaseError as exc:
        raise PreprocessError(f"Invalid SQLite database {path}: {exc}") from exc


def decompress_lzma(source: Path, destination: Path, *, dry_run: bool) -> None:
    if destination.exists():
        validate_sqlite(destination)
        phase(f"{destination.name}: already decompressed and valid")
        return
    if not source.is_file():
        raise PreprocessError(f"Missing compressed source: {source}")
    phase(f"decompressing {source.name} -> {destination.name}")
    if dry_run:
        return
    try:
        with lzma.open(source, "rb") as reader:
            atomic_replace_from_stream(reader, destination)
    except lzma.LZMAError as exc:
        destination.unlink(missing_ok=True)
        raise PreprocessError(f"Could not decompress {source}: {exc}") from exc
    validate_sqlite(destination)


def extract_tar_gz(source: Path, destination: Path, *, dry_run: bool) -> None:
    phase(f"extracting {source.name}")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            safe_relative_path(member.name)
            if member.issym() or member.islnk():
                raise PreprocessError(
                    f"Refusing symbolic/hard link in archive: {member.name!r}"
                )
        archive.extractall(destination, members=members)


def validate_cifar(directory: Path) -> None:
    expected = [
        *(f"data_batch_{index}" for index in range(1, 6)),
        "test_batch",
        "batches.meta",
    ]
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        raise PreprocessError(
            "CIFAR-10 extraction is incomplete; missing: " + ", ".join(missing)
        )


def preprocess_cifar10(root: Path, *, keep_archives: bool, dry_run: bool) -> None:
    raw = root / "cifar10" / "raw"
    archive = raw / "cifar-10-python.tar.gz"
    extracted = raw / "cifar-10-batches-py"

    if extracted.is_dir():
        validate_cifar(extracted)
        phase("cifar10: extracted files already present and valid")
        if not dry_run:
            remove_file(archive, keep_archives)
        return
    if not archive.is_file():
        raise PreprocessError(f"cifar10: expected {archive}")

    extract_tar_gz(archive, raw, dry_run=dry_run)
    if not dry_run:
        validate_cifar(extracted)
        remove_file(archive, keep_archives)
    phase("cifar10: ready")


def preprocess_femnist(root: Path, *, keep_archives: bool, dry_run: bool) -> None:
    raw = root / "femnist" / "raw"
    source = raw / "emnist_all.sqlite.lzma"
    destination = raw / "emnist_all.sqlite"
    decompress_lzma(source, destination, dry_run=dry_run)
    if not dry_run:
        remove_file(source, keep_archives)
    phase("femnist: federated SQLite data ready")


def preprocess_shakespeare(root: Path, *, keep_archives: bool, dry_run: bool) -> None:
    raw = root / "shakespeare" / "raw"
    source = raw / "shakespeare.sqlite.lzma"
    destination = raw / "shakespeare.sqlite"
    decompress_lzma(source, destination, dry_run=dry_run)
    if not dry_run:
        remove_file(source, keep_archives)
    phase("shakespeare: federated SQLite data ready")


def csv_has_consistent_width(path: Path) -> tuple[int, int]:
    """Return (columns, sampled_rows) after a lightweight CSV sanity check."""
    columns: int | None = None
    sampled = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if columns is None:
                    columns = len(row)
                    if columns == 0:
                        raise PreprocessError(f"CSV has no columns: {path}")
                elif len(row) != columns:
                    raise PreprocessError(
                        f"CSV row-width mismatch in {path}: expected {columns}, got {len(row)}"
                    )
                sampled += 1
                if sampled >= 1000:
                    break
    except UnicodeDecodeError as exc:
        raise PreprocessError(f"CSV is not valid UTF-8 text: {path}") from exc
    if columns is None or sampled == 0:
        raise PreprocessError(f"CSV is empty: {path}")
    return columns, sampled


def extract_zip_in_place(archive_path: Path, *, dry_run: bool) -> None:
    target_dir = archive_path.parent
    phase(f"extracting {archive_path.relative_to(target_dir.parent)}")
    if dry_run:
        return
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = safe_relative_path(info.filename)
            target = target_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as reader:
                atomic_replace_from_stream(reader, target)


def find_rar_extractor() -> tuple[str, str] | None:
    """Return (kind, executable) for an available RAR-capable tool.

    Prefer the official unrar implementation when available because some
    N-BaIoT RAR archives use compression methods that 7-Zip may list but
    cannot decode.
    """
    resolved = shutil.which("unrar")
    if resolved:
        return "unrar", resolved

    for executable in ("7z", "7zz", "7za"):
        resolved = shutil.which(executable)
        if resolved:
            return "7z", resolved

    return None


def extract_rar_in_place(archive_path: Path, *, dry_run: bool) -> None:
    phase(f"extracting {archive_path.name}")
    if dry_run:
        return
    extractor = find_rar_extractor()
    if extractor is None:
        raise PreprocessError(
            f"Cannot extract {archive_path.name}: install 7-Zip (7z) or unrar and "
            "ensure it is available on PATH."
        )
    kind, executable = extractor
    if kind == "7z":
        command = [executable, "x", "-y", f"-o{archive_path.parent}", str(archive_path)]
    else:
        command = [executable, "x", "-o+", str(archive_path), str(archive_path.parent)]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-12:])
        raise PreprocessError(
            f"RAR extraction failed for {archive_path} (exit {result.returncode}):\n{tail}"
        )


def iter_attack_archives(raw: Path) -> Iterable[Path]:
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.casefold() in {".rar", ".zip"}:
            yield path


def preprocess_nbaiot(root: Path, *, keep_archives: bool, dry_run: bool) -> None:
    raw = root / "nbaiot" / "raw"
    if not raw.is_dir():
        raise PreprocessError(f"nbaiot: expected directory {raw}")

    # N-BaIoT attack packages may reveal additional archives after extraction.
    # Keep rescanning until the tree contains no archives (unless the user
    # explicitly requested --keep-archives).
    pass_number = 0
    seen_states: set[tuple[str, ...]] = set()
    while True:
        archives = list(iter_attack_archives(raw))
        if not archives:
            if pass_number == 0:
                phase("nbaiot: no compressed attack archives remain")
            break

        relative_archives = tuple(str(path.relative_to(raw)) for path in archives)
        if relative_archives in seen_states and not keep_archives:
            listing = "\n".join(f"  - {name}" for name in relative_archives)
            raise PreprocessError(
                "nbaiot: archive extraction made no progress; remaining archives:\n"
                + listing
            )
        seen_states.add(relative_archives)

        pass_number += 1
        phase(
            f"nbaiot: archive extraction pass {pass_number} "
            f"({len(archives)} archive(s))"
        )

        if dry_run:
            for archive in archives:
                print(f"  would extract: {archive.relative_to(raw)}")
            phase("nbaiot: CSV validation would run after extraction")
            return

        for archive in archives:
            suffix = archive.suffix.casefold()
            if suffix == ".rar":
                extract_rar_in_place(archive, dry_run=False)
            else:
                extract_zip_in_place(archive, dry_run=False)

            remove_file(archive, keep_archives)
            if not keep_archives and archive.exists():
                raise PreprocessError(
                    f"nbaiot: extracted archive could not be removed: {archive}"
                )

        if keep_archives:
            break

    if not keep_archives:
        leftovers = list(iter_attack_archives(raw))
        if leftovers:
            listing = "\n".join(
                f"  - {path.relative_to(raw)} ({path.stat().st_size:,} bytes)"
                for path in leftovers
            )
            raise PreprocessError(
                "nbaiot: preprocessing ended with compressed archives still present:\n"
                + listing
            )

    csv_files = sorted(raw.rglob("*.csv"))
    if not csv_files:
        raise PreprocessError("nbaiot: no CSV files found after archive extraction")

    benign_files = [path for path in csv_files if "benign" in path.name.casefold()]
    attack_files = [path for path in csv_files if "benign" not in path.name.casefold()]
    if not benign_files:
        raise PreprocessError("nbaiot: no benign CSV files found")
    if not attack_files:
        raise PreprocessError("nbaiot: no attack CSV files found")

    phase(f"nbaiot: validating {len(csv_files)} CSV files")
    widths: set[int] = set()
    for path in csv_files:
        columns, _ = csv_has_consistent_width(path)
        widths.add(columns)
    if len(widths) != 1:
        raise PreprocessError(
            f"nbaiot: inconsistent feature widths across CSV files: {sorted(widths)}"
        )

    phase(
        f"nbaiot: ready ({len(benign_files)} benign CSVs, "
        f"{len(attack_files)} attack CSVs, {next(iter(widths))} columns)"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=(*DATASETS, "all"),
        help="One or more installed datasets, or all datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Dataset directory (default: repository-level data/).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep compressed source archives after successful preprocessing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show preprocessing actions without changing files.",
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    if "all" in args.datasets and len(args.datasets) != 1:
        parser.error("Use 'all' alone, or list individual dataset names.")
    return args


def selected_datasets(args: argparse.Namespace) -> Sequence[str]:
    if args.datasets == ["all"]:
        return DATASETS
    return tuple(dict.fromkeys(args.datasets))


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args)
    if not args.data_dir.is_dir():
        raise PreprocessError(
            f"Data directory does not exist: {args.data_dir}. Run install_datasets.py first."
        )

    phase("preprocessing plan")
    print(f"Datasets:      {', '.join(datasets)}")
    print(f"Data root:     {args.data_dir}")
    print(f"Keep archives: {'yes' if args.keep_archives else 'no'}")
    print(f"Dry run:       {'yes' if args.dry_run else 'no'}")

    for dataset in datasets:
        if dataset == "cifar10":
            preprocess_cifar10(
                args.data_dir, keep_archives=args.keep_archives, dry_run=args.dry_run
            )
        elif dataset == "femnist":
            preprocess_femnist(
                args.data_dir, keep_archives=args.keep_archives, dry_run=args.dry_run
            )
        elif dataset == "shakespeare":
            preprocess_shakespeare(
                args.data_dir, keep_archives=args.keep_archives, dry_run=args.dry_run
            )
        elif dataset == "nbaiot":
            preprocess_nbaiot(
                args.data_dir, keep_archives=args.keep_archives, dry_run=args.dry_run
            )

    phase("complete: installed dataset folders are ready for FL loading")


if __name__ == "__main__":
    try:
        main()
    except (PreprocessError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
