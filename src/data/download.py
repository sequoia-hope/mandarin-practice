"""Download AISHELL-3 dataset for Mandarin tone training."""

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm


AISHELL3_URL = "https://www.openslr.org/resources/93/data_aishell3.tgz"
AISHELL3_MD5 = "0c8c1df2ee7e43d44e56a828d8cb0498"


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file using wget with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-c", "-O", str(dest), url],
        check=True,
    )


def extract_tarfile(archive: Path, dest: Path) -> None:
    """Extract tar.gz archive."""
    print(f"Extracting {archive} to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, dest)


def download_aishell3(
    output_dir: str | Path,
    skip_download: bool = False,
    skip_verify: bool = False,
) -> Path:
    """
    Download and extract AISHELL-3 dataset.

    Args:
        output_dir: Directory to store the dataset
        skip_download: Skip download if archive exists
        skip_verify: Skip MD5 verification

    Returns:
        Path to the extracted dataset directory
    """
    output_dir = Path(output_dir)
    archive_path = output_dir / "data_aishell3.tgz"
    extract_dir = output_dir / "aishell3"

    # Check if already extracted
    if extract_dir.exists() and (extract_dir / "train" / "wav").exists():
        print(f"AISHELL-3 already extracted at {extract_dir}")
        return extract_dir

    # Download if needed
    if not archive_path.exists() or not skip_download:
        print(f"Downloading AISHELL-3 from {AISHELL3_URL}...")
        download_file(AISHELL3_URL, archive_path)

    # Verify checksum
    if not skip_verify:
        print("Verifying MD5 checksum...")
        if not verify_md5(archive_path, AISHELL3_MD5):
            raise ValueError("MD5 checksum mismatch! Please re-download the file.")
        print("Checksum verified.")

    # Extract
    extract_tarfile(archive_path, extract_dir)

    return extract_dir


def get_aishell3_splits(dataset_dir: Path) -> dict[str, Path]:
    """Get paths to train/test splits."""
    return {
        "train": dataset_dir / "train",
        "test": dataset_dir / "test",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download AISHELL-3 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MD5 verification",
    )
    args = parser.parse_args()

    download_aishell3(args.output_dir, skip_verify=args.skip_verify)
