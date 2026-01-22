"""Training loop for tone classifier."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .classifier import ToneClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToneDataset(Dataset):
    """
    Dataset for tone classification.

    Loads pre-extracted MMS features and F0 features.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        mms_features_dir: str | Path,
        f0_features_dir: Optional[str | Path] = None,
        use_f0: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            manifest_path: Path to JSON manifest file
            mms_features_dir: Directory with pre-extracted MMS features (.pt files)
            f0_features_dir: Directory with pre-extracted F0 features
            use_f0: Whether to load F0 features
        """
        self.mms_features_dir = Path(mms_features_dir)
        self.f0_features_dir = Path(f0_features_dir) if f0_features_dir else None
        self.use_f0 = use_f0 and f0_features_dir is not None

        # Load manifest
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded {len(self.manifest)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        item = self.manifest[idx]

        # Load MMS features
        mms_path = self.mms_features_dir / f"{item['segment_id']}.pt"
        if mms_path.exists():
            mms_features = torch.load(mms_path)
        else:
            # Fallback to using utt_id
            mms_path = self.mms_features_dir / f"{item['utt_id']}_{item['syllable_idx']:03d}.pt"
            mms_features = torch.load(mms_path) if mms_path.exists() else torch.zeros(2048)

        # Load F0 features
        if self.use_f0 and self.f0_features_dir:
            f0_path = self.f0_features_dir / f"{item['segment_id']}.pt"
            if f0_path.exists():
                f0_features = torch.load(f0_path)
            else:
                f0_features = torch.zeros(3)
        else:
            f0_features = None

        return {
            "mms_features": mms_features,
            "f0_features": f0_features,
            "tone": item["tone"] - 1,  # Convert tone 1-5 to class index 0-4
            "segment_id": item.get("segment_id", idx),
        }


class AudioDataset(Dataset):
    """
    Dataset that loads raw audio and extracts features on-the-fly.

    Slower but doesn't require pre-extraction.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        mms_encoder=None,
        f0_extractor=None,
        sample_rate: int = 16000,
        max_duration: float = 1.0,
    ):
        import soundfile as sf

        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.mms_encoder = mms_encoder
        self.f0_extractor = f0_extractor

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        import librosa

        item = self.manifest[idx]

        # Load audio
        audio, sr = librosa.load(item["audio_path"], sr=self.sample_rate)

        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        return {
            "audio": torch.from_numpy(audio).float(),
            "tone": item["tone"] - 1,  # Convert tone 1-5 to class index 0-4
            "segment_id": item.get("segment_id", idx),
        }


def compute_class_weights(manifest_path: str | Path, num_classes: int = 5) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Uses inverse frequency weighting.
    Handles tone labels 1-5, converting to class indices 0-4.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Count by class index (tone - 1)
    class_counts = {}
    for item in manifest:
        tone = item["tone"]
        class_idx = tone - 1  # Convert tone 1-5 to index 0-4
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    total = sum(class_counts.values())

    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        if 0 <= class_idx < num_classes:
            weights[class_idx] = total / (num_classes * count)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return weights


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    use_f0: bool = True,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        mms_features = batch["mms_features"].to(device)
        f0_features = batch["f0_features"].to(device) if use_f0 and batch["f0_features"] is not None else None
        labels = batch["tone"].to(device)

        optimizer.zero_grad()

        logits = model(mms_features, f0_features)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_f0: bool = True,
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validating"):
        mms_features = batch["mms_features"].to(device)
        f0_features = batch["f0_features"].to(device) if use_f0 and batch["f0_features"] is not None else None
        labels = batch["tone"].to(device)

        logits = model(mms_features, f0_features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train(
    train_manifest: str | Path,
    val_manifest: str | Path,
    mms_features_dir: str | Path,
    f0_features_dir: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints",
    config: Optional[dict] = None,
) -> Path:
    """
    Full training procedure.

    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        mms_features_dir: Directory with MMS features
        f0_features_dir: Directory with F0 features
        output_dir: Output directory for checkpoints
        config: Training configuration dict

    Returns:
        Path to best model checkpoint
    """
    # Default config
    if config is None:
        config = {}

    hidden_dim = config.get("hidden_dim", 256)
    num_classes = config.get("num_classes", 5)
    batch_size = config.get("batch_size", 32)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    num_epochs = config.get("num_epochs", 50)
    patience = config.get("patience", 10)
    use_f0 = config.get("use_f0", True) and f0_features_dir is not None
    dropout = config.get("dropout", 0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = ToneDataset(
        train_manifest,
        mms_features_dir,
        f0_features_dir,
        use_f0=use_f0,
    )
    val_dataset = ToneDataset(
        val_manifest,
        mms_features_dir,
        f0_features_dir,
        use_f0=use_f0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = ToneClassifier(
        input_dim=2048,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_f0=use_f0,
        dropout=dropout,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_manifest).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # Training loop
    best_val_acc = 0.0
    best_model_path = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, use_f0
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_f0)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics["accuracy"])

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model_path = output_dir / f"best_model_acc{best_val_acc:.4f}.pt"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": config,
            }, best_model_path)

            logger.info(f"Saved best model to {best_model_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            break

    logger.info(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_model_path


def load_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> ToneClassifier:
    """Load model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = ToneClassifier(
        input_dim=config.get("input_dim", 2048),
        hidden_dim=config.get("hidden_dim", 256),
        num_classes=config.get("num_classes", 5),
        use_f0=config.get("use_f0", True),
        dropout=config.get("dropout", 0.3),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train tone classifier")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--mms-features", type=str, required=True)
    parser.add_argument("--f0-features", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    train(
        args.train_manifest,
        args.val_manifest,
        args.mms_features,
        args.f0_features,
        args.output,
        config,
    )
