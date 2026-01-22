"""Evaluation metrics and confusion matrix for tone classifier."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .classifier import ToneClassifier, TONE_NAMES
from .train import ToneDataset, load_model


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
) -> dict:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted class indices
        labels: True class indices
        num_classes: Number of classes

    Returns:
        Dict with accuracy, precision, recall, f1 per class and macro-averaged
    """
    # Overall accuracy
    accuracy = (predictions == labels).mean()

    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes)

    for c in range(num_classes):
        true_positives = ((predictions == c) & (labels == c)).sum()
        false_positives = ((predictions == c) & (labels != c)).sum()
        false_negatives = ((predictions != c) & (labels == c)).sum()

        support[c] = (labels == c).sum()

        if true_positives + false_positives > 0:
            precision[c] = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives > 0:
            recall[c] = true_positives / (true_positives + false_negatives)

        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Weighted averages
    weights = support / support.sum()
    weighted_precision = (precision * weights).sum()
    weighted_recall = (recall * weights).sum()
    weighted_f1 = (f1 * weights).sum()

    return {
        "accuracy": float(accuracy),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns:
        (num_classes, num_classes) array where [i,j] = count of true class i predicted as j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for pred, label in zip(predictions, labels):
        cm[label, pred] += 1

    return cm


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> str:
    """Format confusion matrix as string."""
    if class_names is None:
        class_names = [f"T{i+1}" for i in range(len(cm))]

    # Header
    header = "       " + "  ".join(f"{name:>6}" for name in class_names) + "  | Total"
    lines = [header, "-" * len(header)]

    # Rows
    for i, row in enumerate(cm):
        row_total = row.sum()
        row_str = f"{class_names[i]:>6} " + "  ".join(f"{val:>6}" for val in row)
        row_str += f"  | {row_total:>5}"
        lines.append(row_str)

    # Column totals
    col_totals = cm.sum(axis=0)
    total_str = " Total " + "  ".join(f"{val:>6}" for val in col_totals)
    total_str += f"  | {cm.sum():>5}"
    lines.append("-" * len(header))
    lines.append(total_str)

    return "\n".join(lines)


def print_metrics(metrics: dict, class_names: Optional[list[str]] = None) -> str:
    """Format metrics as string."""
    if class_names is None:
        class_names = [f"Tone {i+1}" for i in range(len(metrics["per_class"]["precision"]))]

    lines = [
        f"Overall Accuracy: {metrics['accuracy']:.4f}",
        "",
        "Per-class metrics:",
        f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 55,
    ]

    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<15} "
            f"{metrics['per_class']['precision'][i]:>10.4f} "
            f"{metrics['per_class']['recall'][i]:>10.4f} "
            f"{metrics['per_class']['f1'][i]:>10.4f} "
            f"{int(metrics['per_class']['support'][i]):>10}"
        )

    lines.extend([
        "-" * 55,
        f"{'Macro avg':<15} "
        f"{metrics['macro']['precision']:>10.4f} "
        f"{metrics['macro']['recall']:>10.4f} "
        f"{metrics['macro']['f1']:>10.4f}",
        f"{'Weighted avg':<15} "
        f"{metrics['weighted']['precision']:>10.4f} "
        f"{metrics['weighted']['recall']:>10.4f} "
        f"{metrics['weighted']['f1']:>10.4f}",
    ])

    return "\n".join(lines)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix using matplotlib.

    Args:
        cm: Confusion matrix
        class_names: Class names for labels
        output_path: Path to save figure (shows if None)
        normalize: Whether to normalize rows to percentages
    """
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = [f"Tone {i+1}" for i in range(len(cm))]

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if not normalize else f"{val:.2%}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()

    plt.close()


@torch.no_grad()
def evaluate(
    model: ToneClassifier,
    dataloader: DataLoader,
    device: str = "cuda",
    use_f0: bool = True,
) -> tuple[dict, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained classifier
        dataloader: Test data loader
        device: Device to use
        use_f0: Whether model uses F0

    Returns:
        (metrics_dict, confusion_matrix)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for batch in dataloader:
        mms_features = batch["mms_features"].to(device)
        f0_features = batch["f0_features"].to(device) if use_f0 and batch["f0_features"] is not None else None
        labels = batch["tone"]

        logits = model(mms_features, f0_features)
        _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    metrics = compute_metrics(predictions, labels)
    cm = compute_confusion_matrix(predictions, labels)

    return metrics, cm


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_manifest: str | Path,
    mms_features_dir: str | Path,
    f0_features_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    batch_size: int = 32,
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint
        test_manifest: Path to test manifest
        mms_features_dir: Directory with MMS features
        f0_features_dir: Directory with F0 features
        output_dir: Directory to save results
        batch_size: Batch size for evaluation

    Returns:
        Metrics dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(checkpoint_path, device)

    # Determine if model uses F0
    use_f0 = hasattr(model, "use_f0") and model.use_f0

    # Create dataset
    dataset = ToneDataset(
        test_manifest,
        mms_features_dir,
        f0_features_dir,
        use_f0=use_f0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Evaluate
    metrics, cm = evaluate(model, dataloader, device, use_f0)

    # Print results
    class_names = [f"Tone {i+1}" for i in range(5)]
    print("\n" + print_metrics(metrics, class_names))
    print("\n" + print_confusion_matrix(cm, [f"T{i+1}" for i in range(5)]))

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save confusion matrix
        np.save(output_dir / "confusion_matrix.npy", cm)

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
        plot_confusion_matrix(
            cm, class_names, output_dir / "confusion_matrix_normalized.png",
            normalize=True,
        )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tone classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--mms-features", type=str, required=True)
    parser.add_argument("--f0-features", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint,
        args.test_manifest,
        args.mms_features,
        args.f0_features,
        args.output,
        args.batch_size,
    )
