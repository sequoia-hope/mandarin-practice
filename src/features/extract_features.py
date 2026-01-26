"""Extract encoder features from audio segments.

Supports both MMS and Omnilingual encoders for comparison.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(
    manifest_path: str | Path,
    output_dir: str | Path,
    encoder_type: str = "mms",
    encoder_model: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Extract encoder features from audio segments.

    Args:
        manifest_path: Path to JSON manifest with audio paths
        output_dir: Directory to save extracted features
        encoder_type: 'mms' or 'omnilingual'
        encoder_model: Specific model name (optional)
        batch_size: Batch size for extraction
        device: Torch device
    """
    import librosa

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    logger.info(f"Extracting {encoder_type} features from {len(manifest)} samples")

    # Initialize encoder
    if encoder_type == "mms":
        from .mms_encoder import MMSEncoder

        model_name = encoder_model or "facebook/mms-300m"
        encoder = MMSEncoder(model_name=model_name, device=device)
    elif encoder_type == "omnilingual":
        from .omnilingual_encoder import OmnilingualEncoder

        model_name = encoder_model or "omniASR_W2V_300M"
        encoder = OmnilingualEncoder(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    logger.info(f"Encoder output dimension: {encoder.output_dim}")

    # Save metadata
    metadata = {
        "encoder_type": encoder_type,
        "encoder_model": model_name,
        "output_dim": encoder.output_dim,
        "num_samples": len(manifest),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process in batches
    for i in tqdm(range(0, len(manifest), batch_size), desc=f"Extracting {encoder_type} features"):
        batch_items = manifest[i:i + batch_size]
        batch_audio = []
        batch_ids = []

        for item in batch_items:
            audio_path = item.get("audio_path") or item.get("path")
            segment_id = item.get("segment_id") or f"{item.get('utt_id')}_{item.get('syllable_idx', 0):03d}"

            # Skip if already extracted
            output_path = output_dir / f"{segment_id}.pt"
            if output_path.exists():
                continue

            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                batch_audio.append(audio)
                batch_ids.append(segment_id)
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
                continue

        if not batch_audio:
            continue

        # Pad batch to same length
        max_len = max(len(a) for a in batch_audio)
        padded = np.zeros((len(batch_audio), max_len), dtype=np.float32)
        for j, a in enumerate(batch_audio):
            padded[j, :len(a)] = a

        # Extract features
        with torch.no_grad():
            audio_tensor = torch.from_numpy(padded).to(device)
            features = encoder(audio_tensor)
            pooled = features["pooled"].cpu()

        # Save individual features
        for j, segment_id in enumerate(batch_ids):
            output_path = output_dir / f"{segment_id}.pt"
            torch.save(pooled[j], output_path)

    logger.info(f"Features saved to {output_dir}")


def extract_f0_features(
    manifest_path: str | Path,
    output_dir: str | Path,
    method: str = "pyin",
):
    """
    Extract F0 features from audio segments.

    Args:
        manifest_path: Path to JSON manifest
        output_dir: Directory to save F0 features
        method: F0 extraction method ('pyin' or 'crepe')
    """
    import librosa

    from .f0_extract import F0Extractor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    extractor = F0Extractor(method=method)

    for item in tqdm(manifest, desc="Extracting F0 features"):
        audio_path = item.get("audio_path") or item.get("path")
        segment_id = item.get("segment_id") or f"{item.get('utt_id')}_{item.get('syllable_idx', 0):03d}"

        output_path = output_dir / f"{segment_id}.pt"
        if output_path.exists():
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            result = extractor.extract(audio)

            # Pool F0 features (mean over time)
            f0_features = result["features"].mean(axis=0)
            torch.save(torch.from_numpy(f0_features).float(), output_path)
        except Exception as e:
            logger.warning(f"Failed to extract F0 from {audio_path}: {e}")
            # Save zeros as fallback
            torch.save(torch.zeros(3), output_path)

    logger.info(f"F0 features saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from audio")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["mms", "omnilingual"],
        default="mms",
        help="Encoder type",
    )
    parser.add_argument("--encoder-model", type=str, default=None, help="Specific encoder model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--f0", action="store_true", help="Extract F0 features instead")
    parser.add_argument("--f0-method", type=str, default="pyin", help="F0 method")
    args = parser.parse_args()

    if args.f0:
        extract_f0_features(args.manifest, args.output, args.f0_method)
    else:
        extract_features(
            args.manifest,
            args.output,
            encoder_type=args.encoder,
            encoder_model=args.encoder_model,
            batch_size=args.batch_size,
        )
