#!/usr/bin/env python3
"""Download bounded raw datasets for federated-learning simulations.

This script performs acquisition only. It intentionally does not normalize,
partition, augment, label, tokenize, or construct train/test splits. A separate
preprocessing script should transform the files installed under ``data/``.

Examples:
    python utils/install_datasets.py all --dry-run
    python utils/install_datasets.py all
    python utils/install_datasets.py cifar10 femnist
    python utils/install_datasets.py shakespeare
    python utils/install_datasets.py nbaiot --nbaiot-devices Danmini_Doorbell
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Sequence


GIB = 1024**3
MIB = 1024**2
DATASETS = ("cifar10", "femnist", "shakespeare", "nbaiot")
USER_AGENT = "bounded-fl-raw-data-installer/2.0"


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    expected_bytes: int
    expected_is_approximate: bool = False
    probe_head: bool = True
    sha256: str | None = None
    md5: str | None = None


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str


SOURCES = {
    "cifar10": Source(
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        filename="cifar-10-python.tar.gz",
        expected_bytes=170_498_071,
        md5="c58f30108f718f92721af3b95e74349a",
    ),
    "femnist": Source(
        url="https://storage.googleapis.com/tff-datasets-public/emnist_all.sqlite.lzma",
        filename="emnist_all.sqlite.lzma",
        expected_bytes=170_500_000,
    ),
    "shakespeare": Source(
        url="https://storage.googleapis.com/tff-datasets-public/shakespeare.sqlite.lzma",
        filename="shakespeare.sqlite.lzma",
        expected_bytes=1_329_828,
    ),
    "nbaiot": Source(
        url=(
            "https://archive.ics.uci.edu/static/public/442/"
            "detection+of+iot+botnet+attacks+n+baiot.zip"
        ),
        filename="n-baiot.zip",
        expected_bytes=1_700_000_000,
        expected_is_approximate=True,
        probe_head=False,
    ),
}

NBAIOT_DEFAULT_DEVICES = (
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Provision_PT_737E_Security_Camera",
)


class BudgetError(RuntimeError):
    """Raised when an installation safety limit would be exceeded."""


@dataclass
class Budget:
    max_download: int
    max_work: int
    max_output: int
    downloaded: int = 0
    output_written: int = 0

    def add_download(self, amount: int) -> None:
        self.downloaded += amount
        if self.downloaded > self.max_download:
            raise BudgetError(
                f"Download budget exceeded: {human(self.downloaded)} > "
                f"{human(self.max_download)}. Raise --max-download-gb explicitly."
            )


def phase(message: str) -> None:
    print(f"\n==> {message}")


def human(size: int) -> str:
    if size >= GIB:
        return f"{size / GIB:.2f} GiB"
    return f"{size / MIB:.1f} MiB"


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class Progress:
    """Render a live terminal bar or periodic progress lines for redirected logs."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        approximate: bool = False,
        enabled: bool = True,
    ) -> None:
        self.label = label
        self.total = total
        self.approximate = approximate
        self.enabled = enabled
        self.completed = 0
        self.started = time.monotonic()
        self.last_rendered = 0.0
        self.last_width = 0
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()

    def __enter__(self) -> Progress:
        self.render(force=True)
        return self

    def __exit__(self, *_: object) -> None:
        self.render(force=True)
        if self.enabled and self.interactive:
            print(file=self.stream, flush=True)

    def update(self, amount: int) -> None:
        self.completed += amount
        self.render()

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        interval = 0.15 if self.interactive else 5.0
        if not force and now - self.last_rendered < interval:
            return
        self.last_rendered = now

        elapsed = max(now - self.started, 1e-6)
        rate = self.completed / elapsed
        fraction = min(self.completed / self.total, 1.0) if self.total else 0.0
        percent = 100 * fraction
        eta = (self.total - self.completed) / rate if rate and self.total else None
        eta_text = duration(eta) if eta is not None else "--:--"
        total_marker = "~" if self.approximate else ""
        quantities = (
            f"{human(self.completed)}/{total_marker}{human(self.total)} "
            f"{human(int(rate))}/s ETA {eta_text}"
        )

        if self.interactive:
            bar_width = 28
            filled = int(bar_width * fraction)
            bar = "█" * filled + "░" * (bar_width - filled)
            line = f"{self.label} [{bar}] {percent:6.2f}% {quantities}"
            print(
                "\r" + line.ljust(self.last_width),
                end="",
                file=self.stream,
                flush=True,
            )
            self.last_width = max(self.last_width, len(line))
        else:
            print(
                f"{self.label}: {percent:.2f}% {quantities}",
                file=self.stream,
                flush=True,
            )


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def make_request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT},
    )


def source_size(source: Source) -> tuple[int, bool]:
    """Return source size and whether the value is a documented estimate."""
    if source.probe_head:
        try:
            with urllib.request.urlopen(make_request(source.url, "HEAD"), timeout=30) as response:
                value = response.headers.get("Content-Length")
                if value:
                    return int(value), False
        except (urllib.error.URLError, ValueError):
            pass
    return source.expected_bytes, source.expected_is_approximate


def check_free_space(path: Path, required: int) -> None:
    free = shutil.disk_usage(path).free
    if free < required:
        raise BudgetError(
            f"Insufficient free space: {human(free)} available; "
            f"at least {human(required)} required."
        )


def download(
    source: Source,
    target: Path,
    budget: Budget,
    show_progress: bool = True,
) -> FileRecord:
    declared, declared_is_approximate = source_size(source)
    if budget.downloaded + declared > budget.max_download:
        raise BudgetError(
            f"{source.filename} would exceed the run's "
            f"{human(budget.max_download)} download budget."
        )
    if declared > budget.max_work:
        raise BudgetError(
            f"{source.filename} exceeds --max-work-gb ({human(budget.max_work)})."
        )
    check_free_space(target.parent, declared + 512 * MIB)

    sha256 = hashlib.sha256()
    md5 = hashlib.md5(  # noqa: S324 - official CIFAR compatibility checksum
        usedforsecurity=False
    )
    bytes_written = 0
    with urllib.request.urlopen(make_request(source.url), timeout=120) as response, target.open(
        "wb"
    ) as output:
        content_length = response.headers.get("Content-Length")
        try:
            response_bytes = int(content_length) if content_length else None
        except ValueError:
            response_bytes = None
        progress_total = response_bytes if response_bytes is not None else declared
        with Progress(
            f"Downloading {source.filename}",
            progress_total,
            approximate=response_bytes is None and declared_is_approximate,
            enabled=show_progress,
        ) as progress:
            while True:
                chunk = response.read(MIB)
                if not chunk:
                    break
                budget.add_download(len(chunk))
                output.write(chunk)
                sha256.update(chunk)
                md5.update(chunk)
                bytes_written += len(chunk)
                progress.update(len(chunk))

    if response_bytes is not None and bytes_written != response_bytes:
        raise RuntimeError(
            f"Incomplete download for {source.filename}: received "
            f"{human(bytes_written)} of {human(response_bytes)}."
        )
    if source.sha256 and sha256.hexdigest() != source.sha256:
        raise RuntimeError(f"SHA-256 verification failed for {source.filename}.")
    if source.md5 and md5.hexdigest() != source.md5:
        raise RuntimeError(f"MD5 verification failed for {source.filename}.")
    return FileRecord(source.filename, bytes_written, sha256.hexdigest())


def write_manifest(directory: Path, payload: dict) -> None:
    manifest = {
        "installer_version": 2,
        "stage": "raw_acquisition",
        "requires_preprocessing": True,
        **payload,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_destination(
    data_dir: Path,
    dataset: str,
    force: bool,
    nbaiot_devices: Sequence[str] | None = None,
) -> tuple[Path, Path] | None:
    destination = (data_dir / dataset).resolve()
    if data_dir.resolve() not in destination.parents:
        raise RuntimeError("Refusing to write outside the configured data directory.")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not force:
        if nbaiot_devices is None:
            phase(f"{dataset}: already installed; skipping")
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            installed_devices = manifest.get("selected_devices", [])
        except (OSError, json.JSONDecodeError):
            installed_devices = []
        requested = {device.casefold() for device in nbaiot_devices}
        installed = {
            device.casefold()
            for device in installed_devices
            if isinstance(device, str)
        }
        if requested == installed:
            phase(f"{dataset}: requested devices already installed; skipping")
            return None
        phase(f"{dataset}: requested devices changed; replacing installation")
    staging = data_dir / f".staging-{dataset}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "raw").mkdir(parents=True)
    return staging, destination


def finalize(staging: Path, destination: Path, budget: Budget) -> int:
    size = directory_size(staging)
    projected = budget.output_written + size
    if projected > budget.max_output:
        raise BudgetError(
            f"This run would retain {human(projected)}, above --max-output-gb "
            f"({human(budget.max_output)})."
        )

    backup = destination.with_name(f".{destination.name}.old-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    budget.output_written += size
    if backup.exists():
        shutil.rmtree(backup)
    return size


def install_sources(
    args: argparse.Namespace,
    budget: Budget,
    dataset: str,
    source_keys: Sequence[str],
    description: str,
) -> None:
    prepared = prepare_destination(args.data_dir, dataset, args.force)
    if prepared is None:
        return
    staging, destination = prepared
    phase(f"{dataset}: downloading raw source data")
    records: list[FileRecord] = []
    try:
        for key in source_keys:
            source = SOURCES[key]
            target = staging / "raw" / source.filename
            record = download(
                source,
                target,
                budget,
                show_progress=not args.no_progress,
            )
            records.append(FileRecord(f"raw/{record.path}", record.bytes, record.sha256))
        write_manifest(
            staging,
            {
                "dataset": description,
                "files": [asdict(record) for record in records],
                "sources": [SOURCES[key].url for key in source_keys],
            },
        )
        retained = finalize(staging, destination, budget)
        phase(f"{dataset}: installed {human(retained)}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def safe_zip_parts(name: str) -> tuple[str, ...] | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return tuple(part for part in path.parts if part not in ("", "."))


def selected_nbaiot_path(
    member_name: str,
    devices: Sequence[str],
) -> tuple[str, Path] | None:
    parts = safe_zip_parts(member_name)
    if not parts:
        return None
    for device in devices:
        for index, part in enumerate(parts):
            if part.casefold() == device.casefold():
                return device, Path(device, *parts[index + 1 :])
    return None


def copy_with_hash(
    reader: BinaryIO,
    target: Path,
    progress: Progress | None = None,
) -> FileRecord:
    digest = hashlib.sha256()
    bytes_written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        while True:
            chunk = reader.read(MIB)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            bytes_written += len(chunk)
            if progress is not None:
                progress.update(len(chunk))
    return FileRecord(str(target), bytes_written, digest.hexdigest())


def install_nbaiot(args: argparse.Namespace, budget: Budget) -> None:
    prepared = prepare_destination(
        args.data_dir,
        "nbaiot",
        args.force,
        nbaiot_devices=args.nbaiot_devices,
    )
    if prepared is None:
        return
    staging, destination = prepared
    phase("nbaiot: downloading UCI archive and retaining selected raw device packages")
    try:
        with tempfile.TemporaryDirectory(dir=args.data_dir) as temp_name:
            archive_path = Path(temp_name) / SOURCES["nbaiot"].filename
            archive_record = download(
                SOURCES["nbaiot"],
                archive_path,
                budget,
                show_progress=not args.no_progress,
            )
            records: list[FileRecord] = []
            observed = {
                device: {"benign": False, "attack": False}
                for device in args.nbaiot_devices
            }
            with zipfile.ZipFile(archive_path) as archive:
                selected: list[tuple[zipfile.ZipInfo, str, Path]] = []
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    match = selected_nbaiot_path(member.filename, args.nbaiot_devices)
                    if match is not None:
                        device, relative = match
                        selected.append((member, device, relative))
                if not selected:
                    raise RuntimeError("No requested N-BaIoT device packages were found.")

                selected_bytes = sum(member.file_size for member, _, _ in selected)
                peak_work = archive_record.bytes + selected_bytes
                if peak_work > budget.max_work:
                    raise BudgetError(
                        "The N-BaIoT archive and selected extracted files require "
                        f"approximately {human(peak_work)} together, above "
                        f"--max-work-gb ({human(budget.max_work)})."
                    )
                if budget.output_written + selected_bytes > budget.max_output:
                    raise BudgetError(
                        "Selected N-BaIoT raw files would exceed --max-output-gb."
                    )
                check_free_space(args.data_dir, selected_bytes + 512 * MIB)

                phase(
                    f"nbaiot: extracting {len(selected)} selected files "
                    f"({human(selected_bytes)})"
                )
                with Progress(
                    "Extracting selected N-BaIoT files",
                    selected_bytes,
                    enabled=not args.no_progress,
                ) as progress:
                    for member, device, relative in selected:
                        target = staging / "raw" / relative
                        with archive.open(member) as reader:
                            record = copy_with_hash(reader, target, progress)
                        records.append(
                            FileRecord(
                                str(target.relative_to(staging)),
                                record.bytes,
                                record.sha256,
                            )
                        )
                        lowered = member.filename.casefold()
                        if "benign" in lowered and lowered.endswith(".csv"):
                            observed[device]["benign"] = True
                        if "attack" in lowered and lowered.endswith(
                            (".rar", ".zip", ".csv")
                        ):
                            observed[device]["attack"] = True

            incomplete = [
                device
                for device, roles in observed.items()
                if not roles["benign"] or not roles["attack"]
            ]
            if incomplete:
                raise RuntimeError(
                    "Missing benign or attack packages for: " + ", ".join(incomplete)
                )

        write_manifest(
            staging,
            {
                "dataset": "N-BaIoT selected-device raw packages",
                "source": SOURCES["nbaiot"].url,
                "source_archive": {
                    **asdict(archive_record),
                    "retained": False,
                },
                "selected_devices": list(args.nbaiot_devices),
                "files": [asdict(record) for record in records],
                "note": "Attack packages remain compressed and require preprocessing.",
            },
        )
        retained = finalize(staging, destination, budget)
        phase(f"nbaiot: installed {human(retained)}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=(*DATASETS, "all"),
        help="One or more datasets, or all datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Output directory (default: repository-level data/).",
    )
    parser.add_argument("--force", action="store_true", help="Replace selected datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Show source sizes and exit.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download and extraction progress output.",
    )
    parser.add_argument("--max-download-gb", type=float, default=4.0)
    parser.add_argument("--max-work-gb", type=float, default=4.0)
    parser.add_argument("--max-output-gb", type=float, default=3.0)
    parser.add_argument(
        "--nbaiot-devices",
        nargs="+",
        default=list(NBAIOT_DEFAULT_DEVICES),
        help="N-BaIoT device directories to retain from the full UCI archive.",
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    limits = {
        "--max-download-gb": args.max_download_gb,
        "--max-work-gb": args.max_work_gb,
        "--max-output-gb": args.max_output_gb,
    }
    invalid = [name for name, value in limits.items() if value <= 0]
    if invalid:
        raise SystemExit(f"These parameters must be positive: {', '.join(invalid)}")
    if "all" in args.datasets and len(args.datasets) != 1:
        raise SystemExit("Use 'all' alone, or list individual dataset names.")
    normalized_devices = [device.casefold() for device in args.nbaiot_devices]
    if len(set(normalized_devices)) != len(normalized_devices):
        raise SystemExit("--nbaiot-devices cannot contain duplicates.")


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets == ["all"]:
        return list(DATASETS)
    return list(dict.fromkeys(args.datasets))


def source_keys_for(datasets: Iterable[str]) -> list[str]:
    return list(datasets)


def dry_run(datasets: Sequence[str], max_download: int) -> None:
    phase("preflight: checking raw source sizes")
    total = 0
    for key in source_keys_for(datasets):
        source = SOURCES[key]
        size, approximate = source_size(source)
        marker = "~" if approximate else " "
        print(f"{key:22s} {marker}{human(size):>11s}  {source.url}")
        total += size
    print(f"Expected total: {human(total)}")
    if total > max_download:
        raise BudgetError(
            f"Preflight total exceeds --max-download-gb ({human(max_download)})."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    datasets = selected_datasets(args)
    budget = Budget(
        max_download=int(args.max_download_gb * GIB),
        max_work=int(args.max_work_gb * GIB),
        max_output=int(args.max_output_gb * GIB),
    )
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run(datasets, budget.max_download)
        return

    phase("installation plan")
    print(f"Datasets:    {', '.join(datasets)}")
    print(f"Destination: {args.data_dir}")
    print(
        "Limits:      "
        f"{human(budget.max_download)} download, "
        f"{human(budget.max_work)} working space, "
        f"{human(budget.max_output)} retained output"
    )

    if "cifar10" in datasets:
        install_sources(
            args, budget, "cifar10", ("cifar10",), "CIFAR-10 raw Python archive"
        )
    if "femnist" in datasets:
        install_sources(
            args,
            budget,
            "femnist",
            ("femnist",),
            "Federated EMNIST raw TFF SQLite archive",
        )
    if "shakespeare" in datasets:
        install_sources(
            args,
            budget,
            "shakespeare",
            ("shakespeare",),
            "Federated Shakespeare raw TFF SQLite archive",
        )
    if "nbaiot" in datasets:
        install_nbaiot(args, budget)

    phase(f"complete: raw data written under {args.data_dir}")
    print(f"Downloaded this run: {human(budget.downloaded)}")
    print(f"Retained this run:   {human(budget.output_written)}")


if __name__ == "__main__":
    try:
        main()
    except (BudgetError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
