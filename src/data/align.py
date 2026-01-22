"""Montreal Forced Aligner (MFA) wrapper for syllable-level alignment."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def check_mfa_installed() -> bool:
    """Check if MFA is installed and available."""
    try:
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_mfa_models(
    acoustic_model: str = "mandarin_mfa",
    dictionary: str = "mandarin_pinyin_mfa",
) -> None:
    """
    Download required MFA models for Mandarin.

    Args:
        acoustic_model: Name of acoustic model to download
        dictionary: Name of pronunciation dictionary to download
    """
    print(f"Downloading acoustic model: {acoustic_model}")
    subprocess.run(
        ["mfa", "model", "download", "acoustic", acoustic_model],
        check=True,
    )

    print(f"Downloading dictionary: {dictionary}")
    subprocess.run(
        ["mfa", "model", "download", "dictionary", dictionary],
        check=True,
    )


def prepare_mfa_corpus(
    utterances: list[dict],
    output_dir: str | Path,
) -> Path:
    """
    Prepare corpus in MFA format.

    MFA expects:
    - corpus_dir/
      - speaker1/
        - audio1.wav
        - audio1.txt (transcript)

    Args:
        utterances: List of dicts with 'utt_id', 'audio_path', 'pinyin'
        output_dir: Output directory for prepared corpus

    Returns:
        Path to prepared corpus directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt in tqdm(utterances, desc="Preparing corpus"):
        utt_id = utt["utt_id"]
        audio_path = Path(utt["audio_path"])

        # Extract speaker ID (first 7 chars for AISHELL-3)
        speaker_id = utt_id[:7] if len(utt_id) > 7 else "speaker"
        speaker_dir = output_dir / speaker_id
        speaker_dir.mkdir(exist_ok=True)

        # Copy/symlink audio file
        dest_audio = speaker_dir / f"{utt_id}.wav"
        if not dest_audio.exists():
            if audio_path.exists():
                shutil.copy2(audio_path, dest_audio)

        # Write transcript (pinyin without tone numbers for MFA)
        # MFA uses pinyin without tone markers
        pinyin = utt.get("pinyin", "")
        # Remove tone numbers for MFA dictionary lookup
        pinyin_clean = " ".join(
            "".join(c for c in syl if not c.isdigit())
            for syl in pinyin.split()
        )

        transcript_path = speaker_dir / f"{utt_id}.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(pinyin_clean)

    return output_dir


def run_mfa_alignment(
    corpus_dir: str | Path,
    output_dir: str | Path,
    acoustic_model: str = "mandarin_mfa",
    dictionary: str = "mandarin_pinyin_mfa",
    num_jobs: int = 4,
    clean: bool = True,
) -> Path:
    """
    Run MFA forced alignment on prepared corpus.

    Args:
        corpus_dir: Directory with prepared corpus
        output_dir: Output directory for TextGrid files
        acoustic_model: MFA acoustic model name
        dictionary: MFA dictionary name
        num_jobs: Number of parallel jobs
        clean: Clean up temporary files after alignment

    Returns:
        Path to output directory with TextGrid files
    """
    if not check_mfa_installed():
        raise RuntimeError(
            "MFA not installed. Install with: conda install -c conda-forge montreal-forced-aligner"
        )

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa",
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--num_jobs", str(num_jobs),
        "--output_format", "short_textgrid",
    ]

    if clean:
        cmd.append("--clean")

    print(f"Running MFA alignment: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return output_dir


def run_mfa_validate(
    corpus_dir: str | Path,
    dictionary: str = "mandarin_pinyin_mfa",
) -> dict:
    """
    Validate corpus before alignment.

    Returns dict with validation results.
    """
    corpus_dir = Path(corpus_dir)

    cmd = [
        "mfa",
        "validate",
        str(corpus_dir),
        dictionary,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def align_dataset(
    utterances: list[dict],
    output_dir: str | Path,
    temp_dir: Optional[str | Path] = None,
    num_jobs: int = 4,
) -> Path:
    """
    Full pipeline: prepare corpus and run alignment.

    Args:
        utterances: List of utterance dicts
        output_dir: Final output directory for alignments
        temp_dir: Temporary directory for corpus (cleaned up after)
        num_jobs: Number of parallel jobs

    Returns:
        Path to directory with TextGrid files
    """
    output_dir = Path(output_dir)

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="mfa_corpus_")
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        cleanup_temp = False

    try:
        # Prepare corpus
        corpus_dir = prepare_mfa_corpus(utterances, temp_dir)

        # Run alignment
        alignment_dir = run_mfa_alignment(
            corpus_dir,
            output_dir,
            num_jobs=num_jobs,
        )

        return alignment_dir

    finally:
        if cleanup_temp and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MFA alignment")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--jobs", type=int, default=4, help="Number of jobs")
    parser.add_argument(
        "--download-models", action="store_true", help="Download MFA models first"
    )
    args = parser.parse_args()

    if args.download_models:
        download_mfa_models()

    run_mfa_alignment(args.corpus, args.output, num_jobs=args.jobs)
