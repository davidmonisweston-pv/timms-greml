"""Download the TIMSS 2023 international database.

The IEA's licence is ambiguous about redistributing the public-use data files
(see docs/data-use-policy.md), so this repository never ships the data. It
ships this script, which fetches the files from the official host and records
a checksum so that anyone re-running it can prove they got the same bytes.

Source of truth for the URLs is the TIMSS & PIRLS International Study Center
data page: https://timss2023.org/data/

Only the grade-8 archive is downloaded by default. The archives bundle every
participating education system into one zip; there is no per-country endpoint.

Usage
-----
    uv run python -m timss_greml.download            # grade 8, SPSS
    uv run python -m timss_greml.download --grade 4
    uv run python -m timss_greml.download --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CHECKSUM_DIR = REPO_ROOT / "data" / "checksums"

BASE_URL = "https://timss2023.org/wp-content/uploads/data"

# The User Guide is not data; it is the primary reference for the file naming
# convention and the weight/plausible-value variables, so it is always fetched.
USER_GUIDE = "T23_UG-International-Database.pdf"


@dataclass(frozen=True)
class Archive:
    """One downloadable TIMSS archive."""

    filename: str
    description: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"


ARCHIVES: dict[str, Archive] = {
    "g8": Archive("T23_Data_SPSS_G8.zip", "Grade 8, all systems, SPSS .sav"),
    "g4": Archive("T23_Data_SPSS_G4.zip", "Grade 4, all systems, SPSS .sav"),
    "guide": Archive(USER_GUIDE, "International Database User Guide"),
}

CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Checksum a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def download(archive: Archive, dest_dir: Path, *, force: bool = False) -> Path:
    """Fetch one archive, resuming a partial download where possible."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / archive.filename

    head = requests.head(archive.url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    expected_size = int(head.headers.get("content-length", 0))
    resumable = head.headers.get("accept-ranges", "").lower() == "bytes"

    if dest.exists() and not force:
        if expected_size and dest.stat().st_size == expected_size:
            print(f"  already complete: {archive.filename}")
            return dest
        if not resumable:
            dest.unlink()

    # Leave headroom: the archive has to be unzipped alongside itself later.
    if expected_size and free_bytes(dest_dir) < expected_size * 3:
        raise RuntimeError(
            f"Not enough free disk for {archive.filename} "
            f"({expected_size / 1e9:.1f} GB download, ~3x needed to unzip). "
            f"Free: {free_bytes(dest_dir) / 1e9:.1f} GB."
        )

    start = dest.stat().st_size if (dest.exists() and resumable and not force) else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    mode = "ab" if start else "wb"

    with requests.get(
        archive.url, stream=True, headers=headers, timeout=(30, 300)
    ) as response:
        response.raise_for_status()
        with dest.open(mode) as handle:
            bar = tqdm(
                total=expected_size or None,
                initial=start,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {archive.filename}",
            )
            with bar:
                for block in response.iter_content(chunk_size=CHUNK):
                    handle.write(block)
                    bar.update(len(block))

    actual = dest.stat().st_size
    if expected_size and actual != expected_size:
        raise RuntimeError(
            f"{archive.filename}: expected {expected_size} bytes, got {actual}. "
            "Delete the file and retry."
        )
    return dest


def record_checksum(path: Path) -> str:
    """Write the checksum next to the manifest so downloads are verifiable."""
    CHECKSUM_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CHECKSUM_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    print(f"  checksumming {path.name} ...")
    checksum = sha256_file(path)

    previous = manifest.get(path.name, {}).get("sha256")
    if previous and previous != checksum:
        raise RuntimeError(
            f"{path.name} checksum changed.\n"
            f"  committed: {previous}\n"
            f"  local:     {checksum}\n"
            "The upstream file was revised, or the download is corrupt. "
            "Investigate before analysing - results are not comparable across "
            "different versions of the source data."
        )

    manifest[path.name] = {"sha256": checksum, "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return checksum


def extract(archive_path: Path, dest_dir: Path) -> Path:
    """Unzip into a directory named after the archive."""
    target = dest_dir / archive_path.stem
    if target.exists() and any(target.iterdir()):
        print(f"  already extracted: {target.name}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        for member in tqdm(members, desc=f"  unzip {archive_path.name}", unit="file"):
            zf.extract(member, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grade",
        choices=["4", "8"],
        default="8",
        help="TIMSS population to download (default: 8)",
    )
    parser.add_argument(
        "--no-extract", action="store_true", help="download but do not unzip"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-checksum existing files against the committed manifest",
    )
    args = parser.parse_args(argv)

    wanted = [ARCHIVES["guide"], ARCHIVES[f"g{args.grade}"]]

    if args.verify_only:
        ok = True
        for archive in wanted:
            path = RAW_DIR / archive.filename
            if not path.exists():
                print(f"  MISSING: {archive.filename}")
                ok = False
                continue
            record_checksum(path)
            print(f"  OK: {archive.filename}")
        return 0 if ok else 1

    print(f"Downloading TIMSS 2023 (grade {args.grade}) to {RAW_DIR}")
    print(f"Free disk: {free_bytes(REPO_ROOT) / 1e9:.1f} GB\n")

    for archive in wanted:
        print(f"{archive.description}")
        path = download(archive, RAW_DIR, force=args.force)
        record_checksum(path)
        if path.suffix == ".zip" and not args.no_extract:
            extract(path, RAW_DIR)
        print()

    print("Done. Checksums recorded in data/checksums/manifest.json")
    print(
        "\nCite as: SOURCE: IEA's Trends in International Mathematics and Science "
        "Study - TIMSS 2023. Copyright (c) 2025 International Association for "
        "the Evaluation of Educational Achievement (IEA)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
