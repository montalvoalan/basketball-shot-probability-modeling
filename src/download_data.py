"""Download and verify the public season-level NBA shot-detail files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_REPOSITORY = "cdechoch/nba-data-archive"
DATASET_REVISION = "73fb165135d4d1ef8daaad26fb5b9a29b9d2bdee"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "2021.parquet": {
        "bytes": 2_247_441,
        "sha256": "875bf67a77f0e9d69c26ceae966c28d82de7c31f8f265c518ddb8fbb897fa6c8",
    },
    "2022.parquet": {
        "bytes": 2_245_545,
        "sha256": "ebea41a1c5364f688c33c354b9e9780dfd8ae34c6fdfeadd5d9bc28cad9beba3",
    },
    "2023.parquet": {
        "bytes": 2_254_936,
        "sha256": "1c058cbdb645d4e3ae16ac80d4635d1279d216b6c6490a52bd89929f6d364c54",
    },
    "2024.parquet": {
        "bytes": 2_275_149,
        "sha256": "20cdc43709ba8f4cbe80071c62f377f7c8f5b0b135168cc94545dfba933fba5e",
    },
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    """Check that a downloaded file matches the pinned source artifact."""
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and file_sha256(path) == expected_sha256
    )


def download_file(filename: str, expected_bytes: int, expected_sha256: str) -> None:
    """Download one season atomically and verify its size and checksum."""
    destination = RAW_DATA_DIR / filename
    if is_valid(destination, expected_bytes, expected_sha256):
        print(f"Verified existing file: {destination.relative_to(PROJECT_ROOT)}")
        return

    url = (
        f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
        f"{DATASET_REVISION}/per_season/shotdetail/{filename}?download=true"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "basketball-shot-probability-modeling"})

    print(f"Downloading {filename}...")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

        if not is_valid(temporary, expected_bytes, expected_sha256):
            raise ValueError(f"Downloaded file failed validation: {filename}")

        temporary.replace(destination)
        print(f"Saved and verified: {destination.relative_to(PROJECT_ROOT)}")
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    """Download every selected regular-season file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, metadata in FILES.items():
        download_file(filename, metadata["bytes"], metadata["sha256"])
    print("All selected season files are ready.")


if __name__ == "__main__":
    main()
